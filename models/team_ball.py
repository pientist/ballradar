import torch
import torch.nn as nn

from models.utils import get_params_str, parse_model_params
from set_transformer.model import SetTransformer


class TeamBall(nn.Module):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = ["z_dim", "macro_rnn_dim", "micro_rnn_dim", "n_layers", "dropout"]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.model_type = "macro_regressor"
        self.macro_type = "team_poss"
        self.target_type = params["target_type"]  # options: gk, ball

        self.x_dim = params["n_features"]
        self.z_dim = params["z_dim"]
        self.macro_rnn_dim = params["macro_rnn_dim"]
        self.micro_rnn_dim = params["micro_rnn_dim"]
        self.n_layers = params["n_layers"]
        dropout = params["dropout"]

        self.macro_dim = 2
        self.micro_dim = 4 if self.target_type == "gk" else 2

        self.team1_set_tf = SetTransformer(self.x_dim, self.z_dim)
        self.team2_set_tf = SetTransformer(self.x_dim, self.z_dim)
        self.ppi_fc = nn.Sequential(
            nn.Linear(self.z_dim * 2, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
        )
        self.fpi_fc = SetTransformer(self.x_dim, self.z_dim)

        if params["prev_out_aware"]:
            self.macro_rnn = nn.LSTM(
                input_size=self.z_dim + self.macro_dim,
                hidden_size=self.macro_rnn_dim,
                num_layers=self.n_layers,
                dropout=dropout,
                bidirectional=False,
            )
            self.micro_rnn = nn.LSTM(
                input_size=self.z_dim + self.macro_dim + self.micro_dim,
                hidden_size=self.micro_rnn_dim,
                num_layers=self.n_layers,
                dropout=dropout,
                bidirectional=False,
            )
            if params["bidirectional"]:
                self.macro_rnn_back = nn.LSTM(
                    input_size=self.z_dim,
                    hidden_size=self.macro_rnn_dim,
                    num_layers=self.n_layers,
                    dropout=dropout,
                    bidirectional=False,
                )
                self.micro_rnn_back = nn.LSTM(
                    input_size=self.z_dim,
                    hidden_size=self.micro_rnn_dim,
                    num_layers=self.n_layers,
                    dropout=dropout,
                    bidirectional=False,
                )
        else:
            self.macro_rnn = nn.LSTM(
                input_size=self.z_dim * 2,
                hidden_size=self.macro_rnn_dim,
                num_layers=self.n_layers,
                dropout=dropout,
                bidirectional=params["bidirectional"],
            )
            self.micro_rnn = nn.LSTM(
                input_size=self.z_dim * 2 + self.macro_dim,
                hidden_size=self.micro_rnn_dim,
                num_layers=self.n_layers,
                dropout=dropout,
                bidirectional=params["bidirectional"],
            )

        macro_fc_input_dim = self.macro_rnn_dim * 2 if params["bidirectional"] else self.macro_rnn_dim
        micro_fc_input_dim = self.micro_rnn_dim * 2 if params["bidirectional"] else self.micro_rnn_dim
        self.macro_fc = nn.Sequential(nn.Linear(macro_fc_input_dim, self.macro_dim * 2), nn.GLU())
        self.micro_fc = nn.Sequential(nn.Linear(micro_fc_input_dim, self.micro_dim * 2), nn.GLU())

    def forward(
        self,
        input: torch.Tensor,
        macro_target: torch.Tensor = None,
        micro_target: torch.Tensor = None,
    ) -> torch.Tensor:
        self.macro_rnn.flatten_parameters()
        self.micro_rnn.flatten_parameters()
        if self.params["prev_out_aware"] and self.params["bidirectional"]:
            self.macro_rnn_back.flatten_parameters()
            self.micro_rnn_back.flatten_parameters()

        batch_size = input.size(0)
        seq_len = input.size(1)
        n_players = int(input.size(2) / 2 / self.x_dim)  # 11 for ball regressor and 10 for GK regressor

        input = input.transpose(0, 1).reshape(seq_len * batch_size, -1)  # [time * bs, -1]
        team1_input = input[:, : (self.x_dim * n_players)].reshape(-1, n_players, self.x_dim)  # [time * bs, player, x]
        team2_input = input[:, (self.x_dim * n_players) :].reshape(-1, n_players, self.x_dim)  # [time * bs, player, x]

        team1_z = self.team1_set_tf(team1_input).reshape(seq_len, batch_size, -1)  # [time, bs, z]
        team2_z = self.team2_set_tf(team2_input).reshape(seq_len, batch_size, -1)  # [time, bs, z]
        ppi_z = self.ppi_fc(torch.cat([team1_z, team2_z], dim=-1))  # [time, bs, z]
        fpi_z = self.fpi_fc(input.reshape(-1, n_players * 2, self.x_dim)).reshape(seq_len, batch_size, -1)
        z = torch.cat([ppi_z, fpi_z], dim=-1)  # [time, bs, z * 2]

        if self.params["prev_out_aware"]:
            macro_h_t = torch.zeros(self.n_layers, batch_size, self.macro_rnn_dim).to(input.device)
            macro_c_t = torch.zeros(self.n_layers, batch_size, self.macro_rnn_dim).to(input.device)
            micro_h_t = torch.zeros(self.n_layers, batch_size, self.micro_rnn_dim).to(input.device)
            micro_c_t = torch.zeros(self.n_layers, batch_size, self.micro_rnn_dim).to(input.device)

            if self.params["bidirectional"]:
                macro_h_back_t = torch.zeros(self.n_layers, batch_size, self.macro_rnn_dim).to(input.device)
                macro_c_back_t = torch.zeros(self.n_layers, batch_size, self.macro_rnn_dim).to(input.device)
                micro_h_back_t = torch.zeros(self.n_layers, batch_size, self.micro_rnn_dim).to(input.device)
                micro_c_back_t = torch.zeros(self.n_layers, batch_size, self.micro_rnn_dim).to(input.device)

                macro_h_back_dict = {}
                micro_h_back_dict = {}
                for t in range(seq_len - 1, -1, -1):
                    _, (macro_h_back_t, macro_c_back_t) = self.macro_rnn_back(
                        ppi_z[t].unsqueeze(0), (macro_h_back_t, macro_c_back_t)
                    )
                    _, (micro_h_back_t, micro_c_back_t) = self.micro_rnn_back(
                        ppi_z[t].unsqueeze(0), (micro_h_back_t, micro_c_back_t)
                    )
                    macro_h_back_dict[t] = macro_h_back_t
                    micro_h_back_dict[t] = micro_h_back_t

            macro_pred_list = []
            micro_pred_list = []

            for t in range(seq_len):
                if t == 0:
                    macro_pred_t = torch.zeros(batch_size, self.macro_dim).to(input.device)
                    micro_pred_t = torch.zeros(batch_size, self.micro_dim).to(input.device)

                macro_rnn_input = torch.cat([ppi_z[t], macro_pred_t], dim=-1).unsqueeze(0)
                _, (macro_h_t, macro_c_t) = self.macro_rnn(macro_rnn_input, (macro_h_t, macro_c_t))

                if self.params["bidirectional"]:
                    macro_h_back_t = macro_h_back_dict[t]
                    macro_pred_t = self.macro_fc(torch.cat([macro_h_t[-1], macro_h_back_t[-1]], dim=-1))
                else:
                    macro_pred_t = self.micro_fc(macro_h_t[-1])
                macro_pred_list.append(macro_pred_t)

                micro_rnn_input = torch.cat([ppi_z[t], macro_pred_t, micro_pred_t], dim=-1).unsqueeze(0)
                _, (micro_h_t, micro_c_t) = self.micro_rnn(micro_rnn_input, (micro_h_t, micro_c_t))

                if self.params["bidirectional"]:
                    micro_h_back_t = micro_h_back_dict[t]
                    micro_pred_t = self.micro_fc(torch.cat([micro_h_t[-1], micro_h_back_t[-1]], dim=-1))
                else:
                    micro_pred_t = self.micro_fc(micro_h_t[-1])
                micro_pred_list.append(micro_pred_t)

            macro_out = torch.stack(macro_pred_list, dim=0).transpose(0, 1)  # [bs, time, macro_out]
            micro_out = torch.stack(micro_pred_list, dim=0).transpose(0, 1)  # [bs, time, micro_out]

        else:
            macro_out, _ = self.macro_rnn(z)  # [time, bs, macro_rnn * 2]
            macro_out = self.macro_fc(macro_out)  # [time, bs, macro_out]
            micro_out, _ = self.micro_rnn(torch.cat([z, macro_out], dim=-1))  # [time, bs, micro_rnn * 2]

            macro_out = macro_out.transpose(0, 1)  # [bs, time, macro_out]
            micro_out = self.micro_fc(micro_out).transpose(0, 1)  # [bs, time, micro_out]

        ps = torch.tensor([108, 72]).to(input.device)
        if self.target_type == "gk":
            return torch.cat([macro_out, micro_out[:, :, 0:2] * ps, micro_out[:, :, 2:4] * ps], -1)
        else:
            return torch.cat([macro_out, micro_out * ps], -1)  # [bs, time, macro_out + micro_out]
