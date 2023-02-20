import torch
import torch.nn as nn

from models.utils import get_params_str, parse_model_params
from set_transformer.model import SetTransformer


class TeamBall(nn.Module):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = ["embed_dim", "macro_rnn_dim", "micro_rnn_dim", "n_layers", "dropout"]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.model_type = "macro_regressor"
        self.macro_type = "team_poss"
        self.target_type = params["target_type"]  # options: gk, ball

        self.n_features = params["n_features"]
        self.embed_dim = params["embed_dim"]
        self.macro_rnn_dim = params["macro_rnn_dim"]
        self.micro_rnn_dim = params["micro_rnn_dim"]
        self.n_layers = params["n_layers"]
        dropout = params["dropout"]

        self.macro_dim = 2
        self.micro_dim = 4 if self.target_type == "gk" else 2

        self.team1_set_tf = SetTransformer(in_dim=self.n_features, out_dim=self.embed_dim)
        self.team2_set_tf = SetTransformer(in_dim=self.n_features, out_dim=self.embed_dim)
        self.embed_fc = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        if params["prev_out_aware"]:
            self.macro_rnn = nn.LSTM(
                input_size=self.embed_dim + self.macro_dim,
                hidden_size=self.macro_rnn_dim,
                num_layers=self.n_layers,
                dropout=dropout,
                bidirectional=False,
            )
            self.micro_rnn = nn.LSTM(
                input_size=self.embed_dim + self.macro_dim + self.micro_dim,
                hidden_size=self.micro_rnn_dim,
                num_layers=self.n_layers,
                dropout=dropout,
                bidirectional=False,
            )
            if params["bidirectional"]:
                self.macro_rnn_back = nn.LSTM(
                    input_size=self.embed_dim,
                    hidden_size=self.macro_rnn_dim,
                    num_layers=self.n_layers,
                    dropout=dropout,
                    bidirectional=False,
                )
                self.micro_rnn_back = nn.LSTM(
                    input_size=self.embed_dim,
                    hidden_size=self.micro_rnn_dim,
                    num_layers=self.n_layers,
                    dropout=dropout,
                    bidirectional=False,
                )
        else:
            self.macro_rnn = nn.LSTM(
                input_size=self.embed_dim,
                hidden_size=self.macro_rnn_dim,
                num_layers=self.n_layers,
                dropout=dropout,
                bidirectional=params["bidirectional"],
            )
            self.micro_rnn = nn.LSTM(
                input_size=self.embed_dim + self.macro_dim,
                hidden_size=self.micro_rnn_dim,
                num_layers=self.n_layers,
                dropout=dropout,
                bidirectional=params["bidirectional"],
            )

        macro_fc_input_dim = self.macro_rnn_dim * 2 if params["bidirectional"] else self.macro_rnn_dim
        micro_fc_input_dim = self.micro_rnn_dim * 2 if params["bidirectional"] else self.micro_rnn_dim
        self.macro_fc = nn.Sequential(nn.Linear(macro_fc_input_dim, self.macro_dim * 2), nn.Dropout(dropout), nn.GLU())
        self.micro_fc = nn.Sequential(nn.Linear(micro_fc_input_dim, self.micro_dim * 2), nn.Dropout(dropout), nn.GLU())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.macro_rnn.flatten_parameters()
        self.micro_rnn.flatten_parameters()
        if self.params["bidirectional"]:
            self.macro_rnn_back.flatten_parameters()
            self.micro_rnn_back.flatten_parameters()

        input = input.transpose(0, 1)  # [bs, time, -1] to [time, bs, -1]
        seq_len = input.size(0)
        batch_size = input.size(1)
        n_players = int(input.size(2) / 2 / self.n_features)  # 11 for ball regressor and 10 for GK regressor

        team1_input = input[:, :, : (self.n_features * n_players)].reshape(seq_len * batch_size, -1)
        team2_input = input[:, :, (self.n_features * n_players) :].reshape(seq_len * batch_size, -1)

        team1_embed = self.team1_set_tf(team1_input.reshape(-1, n_players, self.n_features))  # [-1, embed]
        team2_embed = self.team2_set_tf(team2_input.reshape(-1, n_players, self.n_features))  # [-1, embed]
        team1_embed = team1_embed.reshape(seq_len, batch_size, -1)  # [time, bs, embed]
        team2_embed = team2_embed.reshape(seq_len, batch_size, -1)  # [time, bs, embed]
        context_embed = self.embed_fc(torch.cat([team1_embed, team2_embed], dim=-1))  # [time, bs, embed]

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
                        context_embed[t].unsqueeze(0), (macro_h_back_t, macro_c_back_t)
                    )
                    _, (micro_h_back_t, micro_c_back_t) = self.micro_rnn_back(
                        context_embed[t].unsqueeze(0), (micro_h_back_t, micro_c_back_t)
                    )
                    macro_h_back_dict[t] = macro_h_back_t
                    micro_h_back_dict[t] = micro_h_back_t

            macro_pred_list = []
            micro_pred_list = []

            for t in range(seq_len):
                if t == 0:
                    macro_pred_t = torch.zeros(batch_size, self.macro_dim).to(input.device)
                    micro_pred_t = torch.zeros(batch_size, self.micro_dim).to(input.device)

                macro_rnn_input = torch.cat([context_embed[t], macro_pred_t], dim=-1).unsqueeze(0)
                _, (macro_h_t, macro_c_t) = self.macro_rnn(macro_rnn_input, (macro_h_t, macro_c_t))

                if self.params["bidirectional"]:
                    macro_h_back_t = macro_h_back_dict[t]
                    macro_pred_t = self.macro_fc(torch.cat([macro_h_t[-1], macro_h_back_t[-1]], dim=-1))
                else:
                    macro_pred_t = self.micro_fc(macro_h_t[-1])
                macro_pred_list.append(macro_pred_t)

                micro_rnn_input = torch.cat([context_embed[t], macro_pred_t, micro_pred_t], dim=-1).unsqueeze(0)
                _, (micro_h_t, micro_c_t) = self.micro_rnn(micro_rnn_input, (micro_h_t, micro_c_t))

                if self.params["bidirectional"]:
                    micro_h_back_t = micro_h_back_dict[t]
                    micro_pred_t = self.micro_fc(torch.cat([micro_h_t[-1], micro_h_back_t[-1]], dim=-1))
                else:
                    micro_pred_t = self.micro_fc(micro_h_t[-1])
                micro_pred_list.append(micro_pred_t)

            macro_out = torch.stack(macro_pred_list, dim=0).transpose(0, 1)  # [bs, time, macro]
            micro_out = torch.stack(micro_pred_list, dim=0).transpose(0, 1)  # [bs, time, micro]

        else:
            macro_out, _ = self.macro_rnn(context_embed)  # [time, bs, macro_rnn * 2]
            macro_out = self.macro_fc(macro_out)  # [time, bs, macro]
            micro_out, _ = self.micro_rnn(torch.cat([context_embed, macro_out], dim=-1))  # [time, bs, micro_rnn * 2]

            macro_out = macro_out.transpose(0, 1)  # [bs, time, macro]
            micro_out = self.micro_fc(micro_out).transpose(0, 1)  # [bs, time, micro]

        ps = torch.tensor([108, 72]).to(input.device)
        if self.target_type == "gk":
            return torch.cat([macro_out, micro_out[:, :, 0:2] * ps, micro_out[:, :, 2:4] * ps], -1)
        else:
            return torch.cat([macro_out, micro_out * ps], -1)
