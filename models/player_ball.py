import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.utils import get_params_str, parse_model_params
from set_transformer.model import SetTransformer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PlayerBall(nn.Module):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = [
            "n_players",
            "macro_ppe",
            "macro_fpe",
            "macro_fpi",
            "transformer",
            "macro_pe_dim",
            "macro_pi_dim",
            "macro_rnn_dim",
            "micro_pi_dim",
            "micro_rnn_dim",
            "dropout",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.macro_type = params["macro_type"]  # options: player_poss
        self.target_type = params["target_type"]  # options: ball, transition
        self.model_type = "macro_classifier" if self.target_type == "transition" else "macro_regressor"

        n_players = params["n_players"]  # number of players per team (11 in general)
        n_features = params["n_features"]  # number of features per player (6 in general)
        self.macro_dim = n_players * 2 + 4  # number of total players + 4 outside labels (26 in general)
        self.micro_dim = 2

        macro_pe_dim = params["macro_pe_dim"]
        macro_pi_dim = params["macro_pi_dim"]
        macro_rnn_dim = params["macro_rnn_dim"]
        micro_z_dim = params["micro_pi_dim"]
        micro_rnn_dim = params["micro_rnn_dim"]
        n_layers = 2
        dropout = params["dropout"] if "dropout" in params else 0

        if params["macro_ppe"] or params["macro_fpe"] or params["macro_fpi"]:
            macro_z_dim = n_features + 1

            if params["macro_ppe"]:
                self.macro_team1_st = SetTransformer(n_features + 1, macro_pe_dim, embed_type="e")
                self.macro_team2_st = SetTransformer(n_features + 1, macro_pe_dim, embed_type="e")
                self.macro_outside_fc = nn.Linear(n_features + 1, macro_pe_dim)
                macro_z_dim += macro_pe_dim

            if params["macro_fpe"]:
                self.fpe_st = SetTransformer(n_features + 1, macro_pe_dim, embed_type="e")
                macro_z_dim += macro_pe_dim

            if params["macro_fpi"]:
                self.fpi_st = SetTransformer(n_features + 1, macro_pi_dim, embed_type="i")
                macro_z_dim += macro_pi_dim

            if params["transformer"]:
                self.macro_trans_fc = nn.Sequential(nn.Linear(macro_z_dim, macro_rnn_dim * 2), nn.ReLU())
                self.macro_pos_encoder = PositionalEncoding(macro_rnn_dim * 2, dropout)
                encoder_layers = TransformerEncoderLayer(macro_rnn_dim * 2, 4, macro_rnn_dim * 4, dropout)
                self.macro_trans_encoder = TransformerEncoder(encoder_layers, 2)
            else:
                self.macro_rnn = nn.LSTM(
                    input_size=macro_z_dim,
                    hidden_size=macro_rnn_dim,
                    num_layers=n_layers,
                    dropout=dropout,
                    bidirectional=params["bidirectional"],
                )

            macro_fc_dim = macro_rnn_dim * 2 if params["bidirectional"] or params["transformer"] else macro_rnn_dim
            self.macro_fc = nn.Sequential(nn.Linear(macro_fc_dim, 2), nn.GLU())

        else:
            assert not params["transformer"]
            self.macro_rnn = nn.LSTM(
                input_size=(n_features + 1) * self.macro_dim,
                hidden_size=macro_rnn_dim,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=params["bidirectional"],
            )
            macro_fc_dim = macro_rnn_dim * 2 if params["bidirectional"] else macro_rnn_dim
            self.macro_fc = nn.Sequential(nn.Linear(macro_fc_dim, self.macro_dim * 2), nn.GLU())

        self.micro_team1_st = SetTransformer(n_features + macro_rnn_dim * 2 + 1, micro_z_dim, embed_type="i")
        self.micro_team2_st = SetTransformer(n_features + macro_rnn_dim * 2 + 1, micro_z_dim, embed_type="i")
        self.micro_outside_fc = nn.Sequential(
            nn.Linear((n_features + macro_rnn_dim * 2 + 1) * 4, micro_z_dim), nn.ReLU()
        )
        self.micro_embed_fc = nn.Sequential(nn.Linear(micro_z_dim * 3, micro_z_dim), nn.ReLU())

        if params["transformer"]:
            self.micro_trans_fc = nn.Sequential(nn.Linear(micro_z_dim + self.micro_dim, micro_rnn_dim * 2), nn.ReLU())
            self.micro_pos_encoder = PositionalEncoding(micro_rnn_dim * 2, dropout)
            encoder_layers = TransformerEncoderLayer(micro_rnn_dim * 2, 4, micro_rnn_dim * 4, dropout)
            self.micro_trans_encoder = TransformerEncoder(encoder_layers, 2)
        else:
            self.micro_rnn = nn.LSTM(
                input_size=micro_z_dim + self.micro_dim,
                hidden_size=micro_rnn_dim,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=params["bidirectional"],
            )

        micro_fc_dim = micro_rnn_dim * 2 if params["bidirectional"] or params["transformer"] else micro_rnn_dim
        self.micro_fc = nn.Sequential(nn.Linear(micro_fc_dim, self.micro_dim * 2), nn.GLU())

    def forward(
        self,
        input: torch.Tensor,
        macro_target: torch.Tensor = None,
        micro_target: torch.Tensor = None,
        random_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if not self.params["transformer"]:  # for DataParallel employing LSTMs
            self.macro_rnn.flatten_parameters()
            self.micro_rnn.flatten_parameters()

        input = input.transpose(0, 1)  # [bs, time, -1] to [time, bs, -1]
        seq_len = input.size(0)
        batch_size = input.size(1)
        n_players = self.params["n_players"]
        n_features = self.params["n_features"]

        if random_mask is not None:
            random_mask = random_mask.transpose(0, 1)

            if macro_target is not None:
                macro_onehot = nn.functional.one_hot(macro_target, self.macro_dim).transpose(0, 1)
                masked_macro_target = (macro_onehot * random_mask).reshape(seq_len * batch_size, -1).unsqueeze(-1)
            else:
                masked_macro_target = torch.zeros(seq_len * batch_size, self.macro_dim, 1).to(input.device)

            if micro_target is not None:
                masked_micro_target = micro_target.transpose(0, 1) * random_mask
            else:
                masked_micro_target = torch.zeros(seq_len, batch_size, self.micro_dim).to(input.device)

        else:
            masked_macro_target = torch.zeros(seq_len * batch_size, self.macro_dim, 1).to(input.device)
            masked_micro_target = torch.zeros(seq_len, batch_size, self.micro_dim).to(input.device)

        team1_x = input[:, :, : (n_features * n_players)].reshape(seq_len, batch_size, n_players, -1)
        team1_x = team1_x.reshape(seq_len * batch_size, n_players, n_features)  # [time * bs, player, x]
        team1_x_expanded = torch.cat([team1_x, masked_macro_target[:, :n_players]], -1)  # [time * bs, player, x + 1]

        team2_x = input[:, :, (n_features * n_players) : -(n_features * 4)].reshape(seq_len, batch_size, n_players, -1)
        team2_x = team2_x.reshape(seq_len * batch_size, n_players, n_features)  # [time * bs, player, x]
        team2_x_expanded = torch.cat([team2_x, masked_macro_target[:, n_players:-4]], -1)  # [time * bs, player, x + 1]

        outside_x = input[:, :, -(n_features * 4) :].reshape(seq_len, batch_size, 4, -1)
        outside_x = outside_x.reshape(seq_len * batch_size, 4, n_features)  # [time * bs, 4, x]
        outside_x_expanded = torch.cat([outside_x, masked_macro_target[:, -4:]], -1)  # [time * bs, 4, x + 1]

        x = torch.cat([team1_x_expanded, team2_x_expanded, outside_x_expanded], 1)  # [time * bs, macro_out, x + 1]

        if self.params["macro_ppe"] or self.params["macro_fpe"] or self.params["macro_fpi"]:
            macro_z = x

            if self.params["macro_ppe"]:
                team1_z = self.macro_team1_st(team1_x_expanded)  # [time * bs, player, macro_pe]
                team2_z = self.macro_team2_st(team2_x_expanded)  # [time * bs, player, macro_pe]
                outsize_z = self.macro_outside_fc(outside_x_expanded)  # [time * bs, 4, macro_pe]
                ppe_z = torch.cat([team1_z, team2_z, outsize_z], 1)  # [time * bs, macro_out, macro_pe]
                macro_z = torch.cat([x, ppe_z], -1)  # [time * bs, macro_out, x + 1 + macro_pe]

            if self.params["macro_fpe"]:
                fpe_z = self.fpe_st(x)  # [time * bs, macro_out, macro_pe]
                macro_z = torch.cat([macro_z, fpe_z], -1)

            if self.params["macro_fpi"]:
                fpi_z = self.fpi_st(x).unsqueeze(1).expand(-1, self.macro_dim, -1)  # [time * bs, macro_out, macro_pi]
                macro_z = torch.cat([macro_z, fpi_z], -1)

            macro_z = macro_z.reshape(seq_len, batch_size * self.macro_dim, -1)  # [time, bs * macro_out, macro_z]
            if self.params["transformer"]:
                macro_z = self.macro_trans_fc(macro_z)  # [time, bs * macro_out, macro_rnn * 2]
                macro_z = self.macro_pos_encoder(macro_z * math.sqrt(self.params["macro_rnn_dim"]))
                macro_h = self.macro_trans_encoder(macro_z)  # [time, bs * macro_out, macro_rnn * 2]
            else:
                macro_h, _ = self.macro_rnn(macro_z)  # [time, bs * macro_out, macro_rnn * 2]

            macro_h = macro_h.reshape(seq_len * batch_size, self.macro_dim, -1)
            macro_out = self.macro_fc(macro_h)  # [time * bs, macro_out, 1]

        else:  # Use raw inputs (with randomly ordered players) without permutation-equivariant embedding
            macro_z = x.reshape(seq_len, batch_size, -1)  # [time, bs, (x + 1) * macro_out]
            macro_h, _ = self.macro_rnn(macro_z)  # [time, bs, macro_rnn * 2]
            macro_h = macro_h.reshape(seq_len * batch_size, -1)
            macro_out = self.macro_fc(macro_h).unsqueeze(-1)  # [time * bs, macro_out, 1]
            macro_h = macro_h.unsqueeze(1).expand(-1, self.macro_dim, -1)  # [time * bs, macro_out, macro_rnn * 2]

        team1_st_input = torch.cat([team1_x, macro_h[:, :n_players], macro_out[:, :n_players]], -1)
        team2_st_input = torch.cat([team2_x, macro_h[:, n_players:-4], macro_out[:, n_players:-4]], -1)
        outside_fc_input = torch.cat([outside_x, macro_h[:, -4:], macro_out[:, -4:]], -1)
        outside_fc_input = outside_fc_input.reshape(seq_len * batch_size, -1)
        # team_st_input: [time * bs, player, x + macro_rnn * 2 + 1]
        # outside_fc_input: [time * bs, (x + macro_rnn * 2 + 1) * 4]

        micro_team1_z = self.micro_team1_st(team1_st_input)  # [time, bs, micro_z]
        micro_team2_z = self.micro_team2_st(team2_st_input)  # [time, bs, micro_z]
        micro_outside_z = self.micro_outside_fc(outside_fc_input)  # [time, bs, micro_z]

        micro_z = torch.cat([micro_team1_z, micro_team2_z, micro_outside_z], -1)  # [time, bs, micro_z * 3]
        micro_z = self.micro_embed_fc(micro_z).reshape(seq_len, batch_size, -1)  # [time, bs, micro_z]

        micro_z = torch.cat([micro_z, masked_micro_target], -1)  # [time, bs, micro_z + micro_out]
        if self.params["transformer"]:
            micro_z = self.micro_trans_fc(micro_z)  # [time, bs, micro_rnn * 2]
            micro_z = self.micro_pos_encoder(micro_z * math.sqrt(self.params["micro_rnn_dim"]) * 2)
            micro_h = self.micro_trans_encoder(micro_z)  # [time, bs, micro_rnn * 2]
        else:
            micro_h, _ = self.micro_rnn(micro_z)  # [time, bs, micro_rnn * 2]

        macro_out = macro_out.squeeze(-1).reshape(seq_len, batch_size, -1).transpose(0, 1)  # [bs, time, macro_out]
        micro_out = self.micro_fc(micro_h).transpose(0, 1)  # [bs, time, micro_out]

        ps = torch.tensor([108, 72]).to(input.device)
        return torch.cat([macro_out, micro_out * ps], -1)  # [bs, time, macro_out + micro_out]
