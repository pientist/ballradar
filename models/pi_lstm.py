import math

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


class PILSTM(nn.Module):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = ["n_players", "transformer", "context_dim", "rnn_dim", "dropout"]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.target_type = params["target_type"]  # options: "team_poss", "gk", "ball"
        if self.target_type in ["team_poss", "transition"]:
            self.model_type = "classifier"
            self.target_dim = 2
        else:  # self.target_type in ["gk", "ball"]
            self.model_type = "regressor"
            self.target_dim = 4 if self.target_type == "gk" else 2

        n_features = params["n_features"]  # number of features per player (6 in general)
        context_dim = params["context_dim"]
        rnn_dim = params["rnn_dim"]
        dropout = params["dropout"] if "dropout" in params else 0

        self.team1_st = SetTransformer(in_dim=n_features, out_dim=context_dim)
        self.team2_st = SetTransformer(in_dim=n_features, out_dim=context_dim)
        self.context_fc = nn.Sequential(nn.Linear(context_dim * 2, context_dim), nn.ReLU())

        if "transformer" in params and params["transformer"]:
            self.trans_fc = nn.Sequential(nn.Linear(context_dim, rnn_dim * 2), nn.ReLU())
            self.pos_encoder = PositionalEncoding(rnn_dim * 2, dropout)
            encoder_layers = TransformerEncoderLayer(rnn_dim * 2, 4, rnn_dim * 4, dropout)
            self.trans_encoder = TransformerEncoder(encoder_layers, 2)
        else:
            self.rnn = nn.LSTM(
                input_size=context_dim,
                hidden_size=rnn_dim,
                num_layers=2,
                dropout=dropout,
                bidirectional=params["bidirectional"],
            )

        output_fc_dim = rnn_dim * 2 if params["bidirectional"] or params["transformer"] else rnn_dim
        self.output_fc = nn.Sequential(nn.Linear(output_fc_dim, self.target_dim * 2), nn.GLU())

    def forward(self, input: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        if "transformer" not in self.params or not self.params["transformer"]:  # for DataParallel employing LSTM
            self.rnn.flatten_parameters()

        input = input.transpose(0, 1)  # [bs, time, -1] to [time, bs, -1]
        seq_len = input.size(0)
        batch_size = input.size(1)
        n_players = self.params["n_players"]  # number of players per team (10 if target_type == "gk" else 11)
        n_features = self.params["n_features"]  # number of features per player (6 in general)

        team1_input = input[:, :, : (n_features * n_players)].reshape(seq_len * batch_size, -1)
        team2_input = input[:, :, (n_features * n_players) :].reshape(seq_len * batch_size, -1)

        team1_z = self.team1_st(team1_input.reshape(-1, n_players, n_features))  # [time * bs, z]
        team2_z = self.team2_st(team2_input.reshape(-1, n_players, n_features))  # [time * bs, z]
        team1_z = team1_z.reshape(seq_len, batch_size, -1)  # [time, bs, z]
        team2_z = team2_z.reshape(seq_len, batch_size, -1)  # [time, bs, z]
        z = self.context_fc(torch.cat([team1_z, team2_z], -1))  # [time, bs, z]

        if "transformer" in self.params and self.params["transformer"]:
            z = self.trans_fc(z)  # [time, bs, rnn * 2]
            z = self.pos_encoder(z * math.sqrt(self.params["rnn_dim"]) * 2)
            h = self.trans_encoder(z)  # [time, bs, rnn * 2]
        else:
            h, _ = self.rnn(z)  # [time, bs, rnn * 2]

        out = self.output_fc(h).transpose(0, 1)  # [bs, time, out]

        ps = torch.tensor([108, 72]).to(input.device)
        if self.target_type in ["team_poss", "transition"]:
            return out
        elif self.target_type == "gk":
            return torch.cat([out[:, :, 0:2] * ps, out[:, :, 2:4] * ps], dim=2)
        else:
            return out * ps
