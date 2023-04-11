import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import (
    calc_trace_dist,
    get_params_str,
    kld_gauss,
    nll_gauss,
    parse_model_params,
    sample_gauss,
)
from set_transformer.model import SetTransformer


class PIVRNN(nn.Module):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = [
            "n_players",
            "context_dim",
            "rnn_h_dim",
            "n_layers",
            "dropout",
            "vae_h_dim",
            "vae_z_dim",
            "fix_dec_std",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.model_type = "generator"
        self.target_type = params["target_type"]  # options: gk, ball
        self.target_dim = 4 if self.target_type == "gk" else 2

        n_features = params["n_features"]  # number of features per player (6 in general)
        context_dim = params["context_dim"]

        rnn_h_dim = params["rnn_h_dim"]
        n_layers = params["n_layers"]
        dropout = params["dropout"] if "dropout" in params else 0

        vae_h_dim = params["vae_h_dim"]
        vae_z_dim = params["vae_z_dim"]

        self.team1_st = SetTransformer(n_features, context_dim, embed_type="i")
        self.team2_st = SetTransformer(n_features, context_dim, embed_type="i")
        self.context_fc = nn.Sequential(nn.Linear(context_dim * 2, context_dim), nn.ReLU())

        self.rnn_fw = nn.LSTM(context_dim + self.target_dim + vae_z_dim, rnn_h_dim, n_layers, dropout=dropout)
        self.rnn_bw = nn.LSTM(context_dim, rnn_h_dim, n_layers, dropout=dropout)

        self.prior = nn.Sequential(
            nn.Linear(context_dim + rnn_h_dim * 2, vae_h_dim),
            nn.ReLU(),
            nn.Linear(vae_h_dim, vae_h_dim),
            nn.ReLU(),
        )
        self.prior_mean = nn.Linear(vae_h_dim, vae_z_dim)
        self.prior_std = nn.Sequential(nn.Linear(vae_h_dim, vae_z_dim), nn.Softplus())

        self.enc = nn.Sequential(
            nn.Linear(context_dim + self.target_dim + rnn_h_dim * 2, vae_h_dim),
            nn.ReLU(),
            nn.Linear(vae_h_dim, vae_h_dim),
            nn.ReLU(),
        )
        self.enc_mean = nn.Linear(vae_h_dim, vae_z_dim)
        self.enc_std = nn.Sequential(nn.Linear(vae_h_dim, vae_z_dim), nn.Softplus())

        self.dec = nn.Sequential(
            nn.Linear(context_dim + vae_z_dim + rnn_h_dim * 2, vae_h_dim),
            nn.ReLU(),
            nn.Linear(vae_h_dim, vae_h_dim),
            nn.ReLU(),
        )
        self.dec_mean = nn.Linear(vae_h_dim, self.target_dim)
        if not self.params["fix_dec_std"]:
            self.dec_std = nn.Sequential(nn.Linear(vae_h_dim, self.target_dim), nn.Softplus())

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # for DataParallel
        self.rnn_fw.flatten_parameters()
        self.rnn_bw.flatten_parameters()

        loss_tensor = torch.zeros(1, 3).to(input.device)  # [kld_loss, recon_loss, pos_error]
        count = 0

        input = input.transpose(0, 1)  # [bs, time, -1] to [time, bs, -1]
        seq_len = input.size(0)
        batch_size = input.size(1)
        n_players = self.params["n_players"]  # number of players per team (11 in general)
        n_features = self.params["n_features"]  # number of features per player (6 in general)

        ps = torch.tensor([108, 72]).to(input.device)
        target = target.transpose(0, 1) / ps  # [bs, time, -1] to [time, bs, -1]

        team1_x = input[:, :, : (n_features * n_players)].reshape(seq_len, batch_size, n_players, -1)
        team2_x = input[:, :, (n_features * n_players) :].reshape(seq_len, batch_size, n_players, -1)

        team1_context = self.team1_st(team1_x.reshape(seq_len * batch_size, n_players, n_features))
        team2_context = self.team2_st(team2_x.reshape(seq_len * batch_size, n_players, n_features))
        context = torch.cat([team1_context, team2_context], -1)  # [time * bs, context * 2]
        context = self.context_fc(context).reshape(seq_len, batch_size, -1)  # [time, bs, context]

        h_fw = torch.zeros(self.params["n_layers"], batch_size, self.params["rnn_h_dim"]).to(input.device)
        c_fw = torch.zeros(self.params["n_layers"], batch_size, self.params["rnn_h_dim"]).to(input.device)
        h_bw = torch.zeros(self.params["n_layers"], batch_size, self.params["rnn_h_dim"]).to(input.device)
        c_bw = torch.zeros(self.params["n_layers"], batch_size, self.params["rnn_h_dim"]).to(input.device)

        h_bw_dict = {}
        for t in range(seq_len - 1, -1, -1):
            _, (h_bw, c_bw) = self.rnn_bw(context[t].unsqueeze(0), (h_bw, c_bw))
            h_bw_dict[t] = h_bw

        for t in range(seq_len):
            target_t = target[t].clone()  # [bs, target]
            context_t = context[t].clone()  # [bs, context]

            h_t = torch.cat([h_fw[-1], h_bw_dict[t][-1]], -1)  # [bs, rnn_h * 2]

            # prior and encoder
            prior_t = self.prior(torch.cat([context_t, h_t], -1))
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            enc_t = self.enc(torch.cat([context_t, target_t, h_t], -1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # sampling
            z_t = sample_gauss(enc_mean_t, enc_std_t, input.device)  # [bs, vae_z]

            # KLD loss
            loss_tensor[0, 0] += kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

            del prior_t, prior_mean_t, prior_std_t, enc_t, enc_mean_t, enc_std_t

            # decoder
            dec_t = self.dec(torch.cat([context_t, z_t, h_t], -1))
            dec_mean_t = self.dec_mean(dec_t)  # [bs, target]
            if self.params["fix_dec_std"]:
                dec_std_t = (torch.ones(dec_mean_t.shape) * 0.01).to(input.device)
            else:
                dec_std_t = self.dec_std(dec_t)

            # reconstruction loss
            loss_tensor[0, 1] += nll_gauss(dec_mean_t, dec_std_t, target_t, input.device)

            # position error (not used in backprop)
            loss_tensor[0, 2] += calc_trace_dist(dec_mean_t * ps, target_t * ps)
            count += 1

            # recurrence
            _, (h_fw, c_fw) = self.rnn_fw(torch.cat([context_t, target_t, z_t], -1).unsqueeze(0), (h_fw, c_fw))

            del context_t, target_t, z_t, dec_t, dec_mean_t, dec_std_t

        loss_tensor[0, 0] /= batch_size
        loss_tensor[0, 1] /= batch_size
        loss_tensor[0, 2] /= count
        return loss_tensor

    def sample(self, input: torch.Tensor) -> torch.Tensor:
        input = input.transpose(0, 1)
        seq_len = input.size(0)
        batch_size = input.size(1)
        n_players = self.params["n_players"]
        n_features = self.params["n_features"]

        team1_x = input[:, :, : (n_features * n_players)].reshape(seq_len, batch_size, n_players, -1)
        team2_x = input[:, :, (n_features * n_players) :].reshape(seq_len, batch_size, n_players, -1)

        team1_context = self.team1_st(team1_x.reshape(seq_len * batch_size, n_players, n_features))
        team2_context = self.team2_st(team2_x.reshape(seq_len * batch_size, n_players, n_features))
        context = torch.cat([team1_context, team2_context], -1)  # [time * bs, context * 2]
        context = self.context_fc(context).reshape(seq_len, batch_size, -1)  # [time, bs, context]

        h_fw = torch.zeros(self.params["n_layers"], batch_size, self.params["rnn_h_dim"]).to(input.device)
        c_fw = torch.zeros(self.params["n_layers"], batch_size, self.params["rnn_h_dim"]).to(input.device)
        h_bw = torch.zeros(self.params["n_layers"], batch_size, self.params["rnn_h_dim"]).to(input.device)
        c_bw = torch.zeros(self.params["n_layers"], batch_size, self.params["rnn_h_dim"]).to(input.device)

        h_bw_dict = {}
        for t in range(seq_len - 1, -1, -1):
            _, (h_bw, c_bw) = self.rnn_bw(context[t].unsqueeze(0), (h_bw, c_bw))
            h_bw_dict[t] = h_bw

        out = torch.zeros(seq_len, batch_size, self.target_dim).to(input.device)
        for t in range(seq_len):
            context_t = context[t].clone()  # [bs, context]

            h_t = torch.cat([h_fw[-1], h_bw_dict[t][-1]], -1)  # [bs, rnn_h * 2]

            # prior and encoder
            prior_t = self.prior(torch.cat([context_t, h_t], -1))
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling
            z_t = sample_gauss(prior_mean_t, prior_std_t, input.device)

            del prior_t, prior_mean_t, prior_std_t

            # decoder
            dec_t = self.dec(torch.cat([context_t, z_t, h_t], -1))
            out[t] = self.dec_mean(dec_t)  # [bs, target]

            # recurrence
            _, (h_fw, c_fw) = self.rnn_fw(torch.cat([context_t, out[t], z_t], -1).unsqueeze(0), (h_fw, c_fw))

            del context_t, z_t, dec_t

        ps = torch.tensor([108, 72]).to(input.device)
        return out.transpose(0, 1) * ps  # [bs, time, target]
