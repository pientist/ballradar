import torch
import torch.nn as nn

from models.utils import get_params_str, parse_model_params
from set_transformer.model import SetTransformer


class PILSTM(nn.Module):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = ["n_players", "embed_dim", "rnn_dim", "n_layers", "dropout"]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.target_type = params["target_type"]  # options: "team_poss", "transition", "gk", "ball"
        if self.target_type in ["team_poss", "transition"]:
            self.model_type = "classifier"
            self.output_dim = 2
        else:  # self.target_type in ["gk", "ball"]
            self.model_type = "regressor"
            self.output_dim = 4 if self.target_type == "gk" else 2

        self.n_players = params["n_players"]
        self.n_features = params["n_features"]
        self.embed_dim = params["embed_dim"]
        self.rnn_dim = params["rnn_dim"]
        self.n_layers = params["n_layers"]
        dropout = params["dropout"]

        self.team1_set_tf = SetTransformer(in_dim=self.n_features, out_dim=self.embed_dim)
        self.team2_set_tf = SetTransformer(in_dim=self.n_features, out_dim=self.embed_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        if params["prev_out_aware"]:
            self.rnn = nn.LSTM(
                input_size=self.embed_dim + self.output_dim,
                hidden_size=self.rnn_dim,
                num_layers=self.n_layers,
                dropout=dropout,
                bidirectional=False,
            )
            if params["bidirectional"]:
                self.rnn_back = nn.LSTM(
                    input_size=self.embed_dim,
                    hidden_size=self.rnn_dim,
                    num_layers=self.n_layers,
                    dropout=dropout,
                    bidirectional=False,
                )
        else:
            self.rnn = nn.LSTM(
                input_size=self.embed_dim,
                hidden_size=self.rnn_dim,
                num_layers=self.n_layers,
                dropout=dropout,
                bidirectional=params["bidirectional"],
            )

        fc2_input_dim = self.rnn_dim * 2 if params["bidirectional"] else self.rnn_dim
        self.fc2 = nn.Sequential(nn.Linear(fc2_input_dim, self.output_dim * 2), nn.Dropout(dropout), nn.GLU())

    def forward(self, input: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        self.rnn.flatten_parameters()
        if self.params["bidirectional"]:
            self.rnn_back.flatten_parameters()

        input = input.transpose(0, 1)  # [bs, time, -1] to [time, bs, -1]
        if self.training and target is not None:
            target = target.transpose(0, 1)  # [bs, time, -1] to [time, bs, -1]
        seq_len = input.size(0)
        batch_size = input.size(1)
        n_players = int(input.size(2) / 2 / self.n_features)  # 11 for ball regressor and 10 for GK regressor

        team1_input = input[:, :, : (self.n_features * n_players)].reshape(seq_len * batch_size, -1)
        team2_input = input[:, :, (self.n_features * n_players) :].reshape(seq_len * batch_size, -1)

        team1_embed = self.team1_set_tf(team1_input.reshape(-1, n_players, self.n_features))  # [-1, embed]
        team2_embed = self.team2_set_tf(team2_input.reshape(-1, n_players, self.n_features))  # [-1, embed]
        team1_embed = team1_embed.reshape(seq_len, batch_size, -1)  # [time, bs, embed]
        team2_embed = team2_embed.reshape(seq_len, batch_size, -1)  # [time, bs, embed]
        context_embed = self.fc1(torch.cat([team1_embed, team2_embed], dim=-1))  # [time, bs, embed]

        if self.params["prev_out_aware"]:
            h_t = torch.zeros(self.n_layers, batch_size, self.rnn_dim).to(input.device)
            c_t = torch.zeros(self.n_layers, batch_size, self.rnn_dim).to(input.device)

            if self.params["bidirectional"]:
                h_back_t = torch.zeros(self.n_layers, batch_size, self.rnn_dim).to(input.device)
                c_back_t = torch.zeros(self.n_layers, batch_size, self.rnn_dim).to(input.device)

                h_back_dict = {}
                for t in range(seq_len - 1, -1, -1):
                    _, (h_back_t, c_back_t) = self.rnn_back(context_embed[t].unsqueeze(0), (h_back_t, c_back_t))
                    h_back_dict[t] = h_back_t

            pred_list = []

            for t in range(seq_len):
                if t == 0:
                    pred_t = torch.zeros(batch_size, self.output_dim).to(input.device)
                elif self.training and target is not None:
                    pred_t = target[t - 1]  # teacher forcing

                rnn_input = torch.cat([context_embed[t], pred_t], dim=-1).unsqueeze(0)  # [1, bs, embed + out]
                _, (h_t, c_t) = self.rnn(rnn_input, (h_t, c_t))

                if self.params["bidirectional"]:
                    h_back_t = h_back_dict[t]
                    pred_t = self.fc2(torch.cat([h_t[-1], h_back_t[-1]], -1))
                else:
                    pred_t = self.fc2(h_t[-1])  # [bs, out]
                pred_list.append(pred_t)

            out = torch.stack(pred_list, dim=0).transpose(0, 1)  # [bs, time, out]

        else:
            out, _ = self.rnn(context_embed)  # [time, bs, rnn * 2]
            out = self.fc2(out).transpose(0, 1)  # [bs, time, out]

        ps = torch.tensor([108, 72]).to(input.device)
        if self.target_type in ["team_poss", "transition"]:
            return out
        elif self.target_type == "gk":
            return torch.cat([out[:, :, 0:2] * ps, out[:, :, 2:4] * ps], dim=2)
        else:
            return out * ps
