import numpy as np
import torch
import torch.nn as nn

from models.utils import get_params_str, parse_model_params
from set_transformer.model import SetTransformer


class PlayerBall(nn.Module):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = [
            "n_players",
            "macro_ppe",
            "macro_fpe",
            "macro_fpi",
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

        self.n_players = params["n_players"]  # number of players per team (11 in general)
        self.n_components = self.n_players * 2 + 4  # number of total players + 4 outside labels (26 in general)
        self.x_dim = params["n_features"]  # number of features per player (6 in general)

        self.macro_pe_dim = params["macro_pe_dim"]
        self.macro_pi_dim = params["macro_pi_dim"]
        self.macro_rnn_dim = params["macro_rnn_dim"]
        self.micro_pi_dim = params["micro_pi_dim"]
        self.micro_rnn_dim = params["micro_rnn_dim"]
        self.micro_out_dim = 2
        self.n_layers = 2

        dropout = params["dropout"] if "dropout" in params else 0

        assert params["macro_ppe"] or params["macro_fpe"] or params["macro_fpi"]
        macro_rnn_input_dim = self.x_dim + 1

        if params["macro_ppe"]:
            self.macro_team1_st = SetTransformer(self.x_dim + 1, self.macro_pe_dim, embed_type="e")
            self.macro_team2_st = SetTransformer(self.x_dim + 1, self.macro_pe_dim, embed_type="e")
            self.macro_outside_fc = nn.Linear(self.x_dim + 1, self.macro_pe_dim)
            macro_rnn_input_dim += self.macro_pe_dim

        if params["macro_fpe"]:
            self.fpe_st = SetTransformer(self.x_dim + 1, self.macro_pe_dim, embed_type="e")
            macro_rnn_input_dim += self.macro_pe_dim

        if params["macro_fpi"]:
            self.fpi_st = SetTransformer(self.x_dim + 1, self.macro_pi_dim, embed_type="i")
            macro_rnn_input_dim += self.macro_pi_dim

        self.macro_rnn = nn.LSTM(
            input_size=macro_rnn_input_dim,
            hidden_size=self.macro_rnn_dim,
            num_layers=self.n_layers,
            dropout=dropout,
            bidirectional=params["bidirectional"],
        )
        self.macro_fc = nn.Sequential(nn.Linear(self.macro_rnn_dim * 2, 2), nn.GLU())

        self.micro_team1_st = SetTransformer(self.x_dim + self.macro_rnn_dim * 2 + 1, self.micro_pi_dim, embed_type="i")
        self.micro_team2_st = SetTransformer(self.x_dim + self.macro_rnn_dim * 2 + 1, self.micro_pi_dim, embed_type="i")
        self.micro_outside_fc = nn.Sequential(
            nn.Linear((self.x_dim + self.macro_rnn_dim * 2 + 1) * 4, self.micro_pi_dim), nn.ReLU()
        )
        self.micro_embed_fc = nn.Sequential(nn.Linear(self.micro_pi_dim * 3, self.micro_pi_dim), nn.ReLU())

        self.micro_rnn = nn.LSTM(
            input_size=self.micro_pi_dim + self.micro_out_dim,
            hidden_size=self.micro_rnn_dim,
            num_layers=self.n_layers,
            dropout=dropout,
            bidirectional=params["bidirectional"],
        )
        micro_fc_input_dim = self.micro_rnn_dim * 2 if params["bidirectional"] else self.micro_rnn_dim
        self.micro_fc = nn.Sequential(nn.Linear(micro_fc_input_dim, self.micro_out_dim * 2), nn.GLU())

    def forward(
        self,
        input: torch.Tensor,
        macro_target: torch.Tensor = None,
        micro_target: torch.Tensor = None,
        masking_prob: float = 1,  # only used in validation
    ) -> torch.Tensor:
        self.macro_rnn.flatten_parameters()
        self.micro_rnn.flatten_parameters()

        input = input.transpose(0, 1)  # [bs, time, -1] to [time, bs, -1]
        seq_len = input.size(0)
        batch_size = input.size(1)
        n_players = self.params["n_players"]

        # Mask the target trajectories to be used as additional inputs
        if self.training:
            if "masking" in self.params and np.random.choice([True, False], p=[0.5, 0.5]):
                masking_prob = self.params["masking"]
            else:
                masking_prob = 1
        random_mask = (torch.FloatTensor(seq_len, batch_size, 1).uniform_() < 1 - masking_prob).to(input.device)

        if macro_target is not None:
            macro_target_onehot = nn.functional.one_hot(macro_target, self.n_components).transpose(0, 1)
            masked_macro_target = (macro_target_onehot * random_mask).reshape(seq_len * batch_size, -1).unsqueeze(-1)
        else:
            masked_macro_target = torch.zeros(seq_len * batch_size, self.n_components, 1).to(input.device)

        if micro_target is not None:
            masked_micro_target = micro_target.transpose(0, 1) * random_mask
        else:
            masked_micro_target = torch.zeros(seq_len, batch_size, self.micro_out_dim).to(input.device)

        team1_x = input[:, :, : (self.x_dim * n_players)].reshape(seq_len, batch_size, n_players, -1)
        team1_x = team1_x.reshape(seq_len * batch_size, n_players, self.x_dim)  # [time * bs, player, x]
        team1_x_expanded = torch.cat([team1_x, masked_macro_target[:, :n_players]], -1)  # [time * bs, player, x + 1]

        team2_x = input[:, :, (self.x_dim * n_players) : -(self.x_dim * 4)].reshape(seq_len, batch_size, n_players, -1)
        team2_x = team2_x.reshape(seq_len * batch_size, n_players, self.x_dim)  # [time * bs, player, x]
        team2_x_expanded = torch.cat([team2_x, masked_macro_target[:, n_players:-4]], -1)  # [time * bs, player, x + 1]

        outside_x = input[:, :, -(self.x_dim * 4) :].reshape(seq_len, batch_size, 4, -1)
        outside_x = outside_x.reshape(seq_len * batch_size, 4, self.x_dim)  # [time * bs, 4, x]
        outside_x_expanded = torch.cat([outside_x, masked_macro_target[:, -4:]], -1)  # [time * bs, 4, x + 1]

        x = torch.cat([team1_x_expanded, team2_x_expanded, outside_x_expanded], 1)  # [time * bs, comp, x + 1]
        macro_rnn_input = x

        if self.params["macro_ppe"]:
            team1_z = self.macro_team1_st(team1_x_expanded)  # [time * bs, player, macro_pe]
            team2_z = self.macro_team2_st(team2_x_expanded)  # [time * bs, player, macro_pe]
            outsize_z = self.macro_outside_fc(outside_x_expanded)  # [time * bs, 4, macro_pe]
            ppe_z = torch.cat([team1_z, team2_z, outsize_z], dim=1)  # [time * bs, comp, macro_pe]
            macro_rnn_input = torch.cat([x, ppe_z], -1)  # [time * bs, comp, x + 1 + macro_pe]

        if self.params["macro_fpe"]:
            fpe_z = self.fpe_st(x)  # [time * bs, comp, macro_pe]
            macro_rnn_input = torch.cat([macro_rnn_input, fpe_z], -1)

        if self.params["macro_fpi"]:
            fpi_z = self.fpi_st(x).unsqueeze(1).expand(-1, self.n_components, -1)  # [time * bs, comp, macro_pi]
            macro_rnn_input = torch.cat([macro_rnn_input, fpi_z], -1)

        macro_rnn_input = macro_rnn_input.reshape(seq_len, batch_size * self.n_components, -1)
        macro_h, _ = self.macro_rnn(macro_rnn_input)  # [time, bs * comp, macro_rnn * 2]

        macro_h = macro_h.reshape(seq_len * batch_size, self.n_components, -1)  # [time * bs, comp, macro_rnn * 2]
        macro_out = self.macro_fc(macro_h)  # [time * bs, comp, 1]

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

        micro_rnn_input = torch.cat([micro_z, masked_micro_target], dim=-1)  # [time, bs, micro_z + micro_out]
        micro_h, _ = self.micro_rnn(micro_rnn_input)  # [time, bs, micro_rnn * 2]
        micro_out = self.micro_fc(micro_h).transpose(0, 1)  # [bs, time, micro_out]

        macro_out = macro_out.squeeze(-1).reshape(seq_len, batch_size, -1).transpose(0, 1)  # [bs, time, comp]
        ps = torch.tensor([108, 72]).to(input.device)
        return torch.cat([macro_out, micro_out * ps], -1)  # [bs, time, comp + micro_out]
