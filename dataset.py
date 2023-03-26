import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SoccerDataset(Dataset):
    def __init__(
        self,
        data_paths=None,
        target_type="gk",  # options: team_poss, player_poss, gk, ball, transition
        macro_type=None,  # options: team_poss, player_poss
        train=True,
        load_saved=False,
        save_new=False,
        n_features=6,
        window_size=100,
        pitch_size=(108, 72),
        normalize=False,
        target_speed=False,
        flip_pitch=False,
    ):
        self.target_type = target_type
        self.macro_type = macro_type
        self.feature_types = ["_x", "_y", "_vx", "_vy", "_speed", "_accel"]  # total features to save as npy files
        self.n_features = n_features  # number of features among self.feature_types to use in model training
        k = 10 if target_type == "gk" else 11  # number of input players per team

        self.ws = window_size
        self.ps = pitch_size
        self.flip_pitch = flip_pitch

        npz_dir = f"data/{target_type}_pred"

        if load_saved:  # not recommended since direct construction is faster
            assert os.path.exists(npz_dir)
            prefix = "train" if train else "test"
            files = [f for f in os.listdir(npz_dir) if f.startswith(prefix)]
            files.sort()
            assert files

            input_data_list = []
            target_data_list = []

            for f in files:
                npz_data = np.load(f"{npz_dir}/{f}")
                input_data_list.append(npz_data["input"])
                target_data_list.append(npz_data["target"])
                print(f"Dataset loaded from '{npz_dir}/{f}'.")

            input_data = np.concatenate(input_data_list)
            target_data = np.concatenate(target_data_list)

        else:
            assert data_paths is not None

            targets = [target_type]  # "gk" will be modified later
            halfline_x = 0.5 if normalize else self.ps[0] / 2

            input_data_list = []
            target_data_list = []
            if macro_type is not None:
                macro_data_list = []

            for f in tqdm(data_paths, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
                match_traces = pd.read_csv(f, header=0)
                player_cols = [c for c in match_traces.columns if c[0] in ["A", "B"] and c[3:] in self.feature_types]

                if target_type == "transition":
                    trans_flags = (match_traces["player_poss"].isna()).astype(int)

                if macro_type == "player_poss" or target_type == "player_poss":
                    outside_labels = ["OUT-L", "OUT-R", "OUT-B", "OUT-T"]
                    outside_x = [0, self.ps[0], self.ps[0] / 2, self.ps[0] / 2]
                    outside_y = [self.ps[1] / 2, self.ps[1] / 2, 0, self.ps[1]]
                    for i, label in enumerate(outside_labels):
                        match_traces[f"{label}_x"] = outside_x[i]
                        match_traces[f"{label}_y"] = outside_y[i]
                        match_traces[[f"{label}_vx", f"{label}_vy", f"{label}_speed", f"{label}_accel"]] = 0

                if normalize:
                    x_cols = [c for c in match_traces.columns if c.endswith("_x")]
                    y_cols = [c for c in match_traces.columns if c.endswith("_y")]
                    match_traces[x_cols] /= self.ps[0]
                    match_traces[y_cols] /= self.ps[1]

                for phase in match_traces["phase"].unique():
                    if type(phase) == str:  # For GPS-event traces, ignore phases with n_players < 22
                        phase_tuple = [int(i) for i in phase[1:-1].split(",")]
                        if phase_tuple[0] < 0 or phase_tuple[1] < 0:
                            continue

                    phase_traces = match_traces[match_traces["phase"] == phase]

                    team1_gk, team2_gk = SoccerDataset.detect_goalkeepers(phase_traces, halfline_x)
                    team1_code, team2_code = team1_gk[0], team2_gk[0]
                    if target_type == "gk":
                        targets = [team1_gk, team2_gk]

                    input_cols = [c for c in phase_traces[player_cols].dropna(axis=1).columns if c[:3] not in targets]
                    team1_cols = [c for c in input_cols if c.startswith(team1_code)]
                    team2_cols = [c for c in input_cols if c.startswith(team2_code)]
                    input_cols = team1_cols + team2_cols  # Reorder teams so that the left team comes first

                    if min(len(team1_cols), len(team2_cols)) < n_features * k:
                        continue

                    if macro_type == "player_poss" or target_type == "player_poss":
                        input_cols += [f"{label}{t}" for label in outside_labels for t in self.feature_types]
                        poss_labels = [c.split("_")[0] for c in input_cols[::n_features]]
                        poss_dict = dict(zip(poss_labels, np.arange(len(poss_labels))))
                        poss_dict["GOAL-L"] = len(poss_labels) - 4  # same as OUT-L
                        poss_dict["GOAL-R"] = len(poss_labels) - 3  # same as OUT-R

                    if target_type in ["gk", "ball"]:
                        target_cols = [f"{p}{t}" for p in targets for t in ["_x", "_y"]]

                    episodes = [e for e in phase_traces["episode"].unique() if e > 0]
                    for episode in episodes:
                        episode_traces = match_traces[match_traces["episode"] == episode]
                        episode_input = episode_traces[input_cols].values

                        if macro_type == "team_poss":
                            episode_macro = (episode_traces["team_poss"] == team2_code).astype(int).values
                        elif macro_type == "player_poss":
                            player_poss = episode_traces["player_poss"].fillna(method="bfill").fillna(method="ffill")
                            episode_macro = player_poss.map(poss_dict).values

                        if target_type == "transition":
                            episode_target = trans_flags[episode_traces.index].values
                        elif target_type == "team_poss":
                            episode_target = (episode_traces["team_poss"] == team2_code).astype(int).values
                        elif target_type == "player_poss":
                            player_poss = episode_traces["player_poss"].fillna(method="bfill").fillna(method="ffill")
                            episode_target = player_poss.map(poss_dict).values
                        else:  # target_type in ["gk", "ball"]
                            episode_target = episode_traces[target_cols].values
                            if target_speed:
                                x = episode_target[:, 0]
                                y = episode_target[:, 1]
                                vx = np.diff(x, prepend=x[0]) / 0.1
                                vy = np.diff(y, prepend=y[0]) / 0.1
                                speed = np.sqrt(vx**2 + vy**2)
                                episode_target = np.stack([x, y, speed], axis=-1)

                        if len(episode_traces) >= self.ws:
                            for i in range(len(episode_traces) - self.ws + 1):
                                input_data_list.append(episode_input[i : i + self.ws])
                                target_data_list.append(episode_target[i : i + self.ws])
                                if macro_type is not None:
                                    macro_data_list.append(episode_macro[i : i + self.ws])

            input_data = np.stack(input_data_list, axis=0)
            target_data = np.stack(target_data_list, axis=0)
            if macro_type is not None:
                macro_data = np.stack(macro_data_list, axis=0)

            if save_new:  # not recommended since direct construction is faster
                MAX_SAVE_SIZE = 100000
                for i in range(len(input_data) // MAX_SAVE_SIZE + 1):
                    input_slice = input_data[MAX_SAVE_SIZE * i : MAX_SAVE_SIZE * (i + 1)]
                    target_slice = target_data[MAX_SAVE_SIZE * i : MAX_SAVE_SIZE * (i + 1)]
                    file = f"train{i}.npz" if train else f"test{i}.npz"
                    np.savez(f"{npz_dir}/{file}", input=input_slice, target=target_slice)
                    print(f"Dataset saved in '{npz_dir}/{file}'.")

        if normalize:
            self.ps = (1, 1)

        if n_features < 6:
            input_data = input_data.reshape(input_data.shape[0], self.ws, -1, len(self.feature_types))
            input_data = input_data[:, :, :, :n_features].reshape(input_data.shape[0], self.ws, -1)

        if flip_pitch:
            flip_x = np.random.choice(2, (input_data.shape[0], 1, 1))
            flip_y = np.random.choice(2, (input_data.shape[0], 1, 1))
            valid_dim = n_features * (k * 2)  # valid input dimension only including player features

            # (ref, mul) = (ps, -1) if flip == 1 else (0, 1)
            ref_x = flip_x * self.ps[0]
            ref_y = flip_y * self.ps[1]
            mul_x = 1 - flip_x * 2
            mul_y = 1 - flip_y * 2

            # flip x and y
            input_data[:, :, 0:valid_dim:n_features] = input_data[:, :, 0:valid_dim:n_features] * mul_x + ref_x
            input_data[:, :, 1:valid_dim:n_features] = input_data[:, :, 1:valid_dim:n_features] * mul_y + ref_y
            if target_type == "gk":
                target_data[:, :, 0::2] = target_data[:, :, 0::2] * mul_x + ref_x
                target_data[:, :, 1::2] = target_data[:, :, 1::2] * mul_y + ref_y
            elif target_type == "ball":
                target_data[:, :, [0]] = target_data[:, :, [0]] * mul_x + ref_x
                target_data[:, :, [1]] = target_data[:, :, [1]] * mul_y + ref_y

            # flip vx and vy
            if n_features > 2:
                input_data[:, :, 2:valid_dim:n_features] = input_data[:, :, 2:valid_dim:n_features] * mul_x
                input_data[:, :, 3:valid_dim:n_features] = input_data[:, :, 3:valid_dim:n_features] * mul_y

            # if flip_x == 1, reorder team1 and team2 features
            team1_input = input_data[:, :, : n_features * k]
            team2_input = input_data[:, :, n_features * k : valid_dim]
            if macro_type == "player_poss" or target_type == "player_poss":
                outside_input = input_data[:, :, valid_dim:]
                input_permuted = np.concatenate([team2_input, team1_input, outside_input], -1)
            else:
                input_permuted = np.concatenate([team2_input, team1_input], -1)
            input_data = np.where(flip_x, input_permuted, input_data)

            if macro_type == "team_poss":
                # if flip_x == 1, switch team1 (0) and team2 (1)
                macro_data = np.where(flip_x.squeeze(-1), 1 - macro_data, macro_data)

            elif macro_type == "player_poss":
                # if flip_x == 1, switch team1 (0-10) and team2 (11-21), and switch OUT-L (22) and OUT-R (23)
                team1_permuted = np.where(macro_data < k, macro_data + k, 0)
                team2_permuted = np.where((macro_data >= k) & (macro_data < k * 2), macro_data - k, 0)
                macro_invariant = np.where(np.isin(macro_data, [k * 2 + 2, k * 2 + 3]), macro_data, 0)
                out_l_to_r = np.where(macro_data == k * 2, k * 2 + 1, 0)
                out_r_to_l = np.where(macro_data == k * 2 + 1, k * 2, 0)
                macro_permuted = team1_permuted + team2_permuted + macro_invariant + out_l_to_r + out_r_to_l
                macro_data = np.where(flip_x.squeeze(-1), macro_permuted, macro_data)

                # if flip_y == 1, switch OUT-B (24) and OUT-T (25)
                macro_invariant = np.where(macro_data < k * 2 + 2, macro_data, 0)
                out_b_to_t = np.where(macro_data == k * 2 + 2, k * 2 + 3, 0)
                out_t_to_b = np.where(macro_data == k * 2 + 3, k * 2 + 2, 0)
                macro_permuted = macro_invariant + out_b_to_t + out_t_to_b
                macro_data = np.where(flip_y.squeeze(-1), macro_permuted, macro_data)

            if target_type == "team_poss":
                # if flip_x == 1, switch team1 (0) and team2 (1)
                target_data = np.where(flip_x.squeeze(-1), 1 - target_data, target_data)

            elif target_type == "player_poss":
                # if flip_x == 1, switch team1 (0-10) and team2 (11-21), and switch OUT-L (22) and OUT-R (23)
                team1_permuted = np.where(target_data < k, target_data + k, 0)
                team2_permuted = np.where((target_data >= k) & (target_data < k * 2), target_data - k, 0)
                target_invariant = np.where(np.isin(target_data, [k * 2 + 2, k * 2 + 3]), target_data, 0)
                out_l_to_r = np.where(target_data == k * 2, k * 2 + 1, 0)
                out_r_to_l = np.where(target_data == k * 2 + 1, k * 2, 0)
                target_permuted = team1_permuted + team2_permuted + target_invariant + out_l_to_r + out_r_to_l
                target_data = np.where(flip_x.squeeze(-1), target_permuted, target_data)

                # if flip_y == 1, switch OUT-B (24) and OUT-T (25)
                target_invariant = np.where(target_data < k * 2 + 2, target_data, 0)
                out_b_to_t = np.where(target_data == k * 2 + 2, k * 2 + 3, 0)
                out_t_to_b = np.where(target_data == k * 2 + 3, k * 2 + 2, 0)
                target_permuted = target_invariant + out_b_to_t + out_t_to_b
                target_data = np.where(flip_y.squeeze(-1), target_permuted, target_data)

            elif target_type == "gk":
                # if flip_x == 1, switch team1_gk and team2_gk
                target_permuted = np.concatenate([target_data[:, :, 2:], target_data[:, :, :2]], -1)
                target_data = np.where(flip_x, target_permuted, target_data)

        self.input_data = torch.FloatTensor(input_data)
        if macro_type in ["team_poss", "player_poss"]:
            self.macro_data = torch.LongTensor(macro_data)
        if target_type in ["transition", "team_poss", "player_poss"]:
            self.target_data = torch.LongTensor(target_data)
        else:  # target_type in ["gk", "ball"]
            self.target_data = torch.FloatTensor(target_data)

    def __getitem__(self, i):
        if self.macro_type is None:
            return self.input_data[i], self.target_data[i]
        else:
            return self.input_data[i], self.macro_data[i], self.target_data[i]

    def __len__(self):
        return len(self.input_data)

    @staticmethod
    def detect_goalkeepers(traces: pd.DataFrame, halfline_x=54):
        a_x_cols = [c for c in traces.columns if c.startswith("A") and c.endswith("_x")]
        b_x_cols = [c for c in traces.columns if c.startswith("B") and c.endswith("_x")]

        a_gk = (traces[a_x_cols].mean() - halfline_x).abs().idxmax()[:3]
        b_gk = (traces[b_x_cols].mean() - halfline_x).abs().idxmax()[:3]

        a_gk_mean_x = traces[f"{a_gk}_x"].mean()
        b_gk_mean_y = traces[f"{b_gk}_x"].mean()

        return (a_gk, b_gk) if a_gk_mean_x < b_gk_mean_y else (b_gk, a_gk)


if __name__ == "__main__":
    dir = "data/metrica_traces"
    filepaths = [f"{dir}/{f}" for f in os.listdir(dir) if f.endswith(".csv")]
    filepaths.sort()
    dataset = SoccerDataset(filepaths[-1:], target_type="gk", train=False, save=False)
    print(dataset[10000][2])
