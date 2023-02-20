import torch
import torch.nn as nn

from models.utils import get_params_str, parse_model_params
from set_transformer.model import SetTransformer


class PELSTM(nn.Module):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = ["n_players", "embed_dim", "rnn_dim", "n_layers", "dropout"]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.target_type = params["target_type"]  # player_poss
        self.model_type = "classifier"

        self.n_players = params["n_players"]  # number of players per team (11 in general)
        self.n_components = self.n_players * 2 + 4  # number of total players + 4 outside labels (26 in general)
        self.x_dim = params["n_features"]  # number of features per player (6 in general)
        self.z_dim = params["embed_dim"]
        self.h_dim = params["rnn_dim"]
        self.n_layers = params["n_layers"]
        dropout = params["dropout"] if "dropout" in params else 0

        self.team1_st_fw = SetTransformer(self.x_dim + self.h_dim, self.z_dim, embed_type="equivariant")
        self.team2_st_fw = SetTransformer(self.x_dim + self.h_dim, self.z_dim, embed_type="equivariant")
        self.outside_fc_fw = nn.Sequential(
            nn.Linear(self.x_dim + self.h_dim, self.z_dim), nn.Dropout(dropout), nn.ReLU()
        )
        self.rnn_fw = nn.LSTM(
            input_size=self.x_dim + self.z_dim,
            hidden_size=self.h_dim,
            num_layers=self.n_layers,
            dropout=dropout,
            bidirectional=False,
        )

        if not params["bidirectional"]:
            self.fc = nn.Sequential(nn.Linear(self.h_dim, 2), nn.Dropout(dropout), nn.GLU())

        else:
            self.team1_st_bw = SetTransformer(self.x_dim + self.h_dim, self.z_dim, embed_type="equivariant")
            self.team2_st_bw = SetTransformer(self.x_dim + self.h_dim, self.z_dim, embed_type="equivariant")
            self.outside_fc_bw = nn.Sequential(
                nn.Linear(self.x_dim + self.h_dim, self.z_dim), nn.Dropout(dropout), nn.ReLU()
            )
            self.rnn_bw = nn.LSTM(
                input_size=self.x_dim + self.z_dim,
                hidden_size=self.h_dim,
                num_layers=self.n_layers,
                dropout=dropout,
                bidirectional=False,
            )
            self.fc = nn.Sequential(nn.Linear(self.h_dim * 2, 2), nn.Dropout(dropout), nn.GLU())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.rnn_fw.flatten_parameters()
        if self.params["bidirectional"]:
            self.rnn_bw.flatten_parameters()

        input = input.transpose(0, 1)  # [bs, time, -1] to [time, bs, -1]
        seq_len = input.size(0)
        batch_size = input.size(1)
        n_players = self.params["n_players"]

        team1_x = input[:, :, : (self.x_dim * n_players)].reshape(seq_len, batch_size, n_players, -1)
        team2_x = input[:, :, (self.x_dim * n_players) : -(self.x_dim * 4)].reshape(seq_len, batch_size, n_players, -1)
        outside_x = input[:, :, -(self.x_dim * 4) :].reshape(seq_len, batch_size, 4, -1)

        h_fw_t = torch.zeros(self.n_layers, batch_size, self.n_components, self.h_dim).to(input.device)
        c_fw_t = torch.zeros(self.n_layers, batch_size, self.n_components, self.h_dim).to(input.device)
        h_bw_t = torch.zeros(self.n_layers, batch_size, self.n_components, self.h_dim).to(input.device)
        c_bw_t = torch.zeros(self.n_layers, batch_size, self.n_components, self.h_dim).to(input.device)

        h_fw_dict = dict()
        h_bw_dict = dict()
        pred_list = []

        for t in range(seq_len):
            team1_x_t = team1_x[t]  # [bs, player, x]
            team2_x_t = team2_x[t]  # [bs, player, x]
            outside_x_t = outside_x[t]  # [bs, 4, x]

            team1_z_t = self.team1_st_fw(torch.cat([team1_x_t, h_fw_t[-1, :, :n_players]], -1))  # [bs, player, z]
            team2_z_t = self.team2_st_fw(torch.cat([team2_x_t, h_fw_t[-1, :, n_players:-4]], -1))  # [bs, player, z]
            outside_z_t = self.outside_fc_fw(torch.cat([outside_x_t, h_fw_t[-1, :, -4:]], -1))  # [bs, 4, z]

            x_t = torch.cat([team1_x_t, team2_x_t, outside_x_t], 1)  # [bs, comp, x]
            z_t = torch.cat([team1_z_t, team2_z_t, outside_z_t], 1)  # [bs, comp, z]

            rnn_input = torch.cat([x_t, z_t], -1)
            rnn_input = rnn_input.reshape(batch_size * self.n_components, -1).unsqueeze(0)  # [bs * comp, x + z]
            h_fw_t = h_fw_t.reshape(self.n_layers, batch_size * self.n_components, self.h_dim)  # [2, bs * comp, h]
            c_fw_t = c_fw_t.reshape(self.n_layers, batch_size * self.n_components, self.h_dim)  # [2, bs * comp, h]
            _, (h_fw_t, c_fw_t) = self.rnn_fw(rnn_input, (h_fw_t, c_fw_t))
            h_fw_dict[t] = h_fw_t[-1]

            if not self.params["bidirectional"]:
                pred_t = self.fc(h_fw_t[-1]).reshape(batch_size, self.n_components, 1).squeeze(2)  # [bs, comp]
                pred_list.append(pred_t)

            h_fw_t = h_fw_t.reshape(self.n_layers, batch_size, self.n_components, self.h_dim)
            c_fw_t = c_fw_t.reshape(self.n_layers, batch_size, self.n_components, self.h_dim)

        if self.params["bidirectional"]:
            for t in range(seq_len - 1, -1, -1):
                team1_x_t = team1_x[t]  # [bs, player, x]
                team2_x_t = team2_x[t]  # [bs, player, x]
                outside_x_t = outside_x[t]  # [bs, 4, x]

                team1_z_t = self.team1_st_bw(torch.cat([team1_x_t, h_bw_t[-1, :, :n_players]], -1))  # [bs, player, z]
                team2_z_t = self.team2_st_bw(torch.cat([team2_x_t, h_bw_t[-1, :, n_players:-4]], -1))  # [bs, player, z]
                outside_z_t = self.outside_fc_bw(torch.cat([outside_x_t, h_bw_t[-1, :, -4:]], -1))  # [bs, 4, z]

                x_t = torch.cat([team1_x_t, team2_x_t, outside_x_t], 1)  # [bs, comp, x]
                z_t = torch.cat([team1_z_t, team2_z_t, outside_z_t], 1)  # [bs, comp, z]

                rnn_input = torch.cat([x_t, z_t], -1)
                rnn_input = rnn_input.reshape(batch_size * self.n_components, -1).unsqueeze(0)  # [bs * comp, x + z]
                h_bw_t = h_bw_t.reshape(self.n_layers, batch_size * self.n_components, self.h_dim)  # [2, bs * comp, h]
                c_bw_t = c_bw_t.reshape(self.n_layers, batch_size * self.n_components, self.h_dim)  # [2, bs * comp, h]
                _, (h_bw_t, c_bw_t) = self.rnn_bw(rnn_input, (h_bw_t, c_bw_t))
                h_bw_dict[t] = h_bw_t[-1]

                h_bw_t = h_bw_t.reshape(self.n_layers, batch_size, self.n_components, self.h_dim)
                c_bw_t = c_bw_t.reshape(self.n_layers, batch_size, self.n_components, self.h_dim)

            for t in range(seq_len):
                h_t = torch.cat([h_fw_dict[t], h_bw_dict[t]], -1)  # [bs * comp, h * 2]
                pred_t = self.fc(h_t).reshape(batch_size, self.n_components, 1).squeeze(2)  # [bs, comp]
                pred_list.append(pred_t)

        return torch.stack(pred_list, 0).transpose(0, 1)  # [bs, time, comp]
