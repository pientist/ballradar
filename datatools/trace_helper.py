import os
import sys
from collections import Counter

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib.colors as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
import torch.nn as nn
from matplotlib import animation
from scipy.ndimage import shift
from tqdm import tqdm

from dataset import SoccerDataset
from models import PIVRNN, PlayerBall
from models.utils import calc_class_acc, calc_real_loss, calc_trace_dist


class TraceHelper:
    def __init__(self, traces: pd.DataFrame, events: pd.DataFrame = None, pitch_size: tuple = (108, 72)):
        self.traces = traces.dropna(axis=1, how="all").copy()
        self.events = events
        self.pitch_size = pitch_size

        self.team1_players = [c[:-2] for c in self.traces.columns if c.startswith("A") and c.endswith("_x")]
        self.team2_players = [c[:-2] for c in self.traces.columns if c.startswith("B") and c.endswith("_x")]

        self.team1_cols = np.array([TraceHelper.player_to_cols(p) for p in self.team1_players]).flatten().tolist()
        self.team2_cols = np.array([TraceHelper.player_to_cols(p) for p in self.team2_players]).flatten().tolist()

        self.phase_records = None

    @staticmethod
    def player_to_cols(p):
        return [f"{p}_x", f"{p}_y", f"{p}_vx", f"{p}_vy", f"{p}_speed", f"{p}_accel"]

    def calc_single_player_running_features(self, p: str, remove_outliers=True, smoothing=True):
        if remove_outliers:
            MAX_SPEED = 12
            MAX_ACCEL = 8

        if smoothing:
            W_LEN = 11
            P_ORDER = 2

        x = self.traces[f"{p}_x"].dropna()
        y = self.traces[f"{p}_y"].dropna()
        vx = np.diff(x.values, prepend=x.iloc[0]) / 0.1
        vy = np.diff(y.values, prepend=y.iloc[0]) / 0.1

        if remove_outliers:
            speeds = np.sqrt(vx**2 + vy**2)
            is_speed_outlier = speeds > MAX_SPEED
            is_accel_outlier = np.abs(np.diff(speeds, append=speeds[-1]) / 0.1) > MAX_ACCEL
            is_outlier = is_speed_outlier | is_accel_outlier | shift(is_accel_outlier, 1, cval=True)
            vx = pd.Series(np.where(is_outlier, np.nan, vx)).interpolate(limit_direction="both").values
            vy = pd.Series(np.where(is_outlier, np.nan, vy)).interpolate(limit_direction="both").values

        if smoothing:
            vx = signal.savgol_filter(vx, window_length=W_LEN, polyorder=P_ORDER)
            vy = signal.savgol_filter(vy, window_length=W_LEN, polyorder=P_ORDER)

        speeds = np.sqrt(vx**2 + vy**2)
        accels = np.diff(speeds, append=speeds[-1]) / 0.1

        if smoothing:
            accels = signal.savgol_filter(accels, window_length=W_LEN, polyorder=P_ORDER)

        self.traces.loc[x.index, TraceHelper.player_to_cols(p)[2:]] = np.stack([vx, vy, speeds, accels]).round(6).T

    def calc_running_features(self, remove_outliers=True, smoothing=True):
        for p in tqdm(self.team1_players + self.team2_players, desc="Calculating running features"):
            self.calc_single_player_running_features(p, remove_outliers, smoothing)

        data_cols = self.team1_cols + self.team2_cols
        if "ball_x" in self.traces.columns:
            data_cols += ["ball_x", "ball_y"]
        meta_cols = self.traces.columns[: len(self.traces.columns) - len(data_cols)].tolist()
        self.traces = self.traces[meta_cols + data_cols]

    @staticmethod
    def ffill_transition(team_poss):
        team = team_poss.iloc[0]
        nans = Counter(team_poss)[0] // 2
        team_poss.iloc[:-nans] = team
        return team_poss.replace({0: np.nan, 1: "A", 2: "B"})

    def find_gt_team_poss(self, player_poss_col="player_poss"):
        self.traces["team_poss"] = self.traces[player_poss_col].fillna(method="bfill").fillna(method="ffill")
        self.traces["team_poss"] = self.traces["team_poss"].apply(lambda x: x[0])

        # team_poss_dict = {"T": np.nan, "O": 0, "A": 1, "B": 2}
        # team_poss = self.traces[player_poss_col].fillna("T").apply(lambda x: x[0]).map(team_poss_dict)
        # poss_ids = (team_poss.diff().fillna(0) * team_poss).cumsum()
        # team_poss = team_poss.groupby(poss_ids, group_keys=True).apply(TraceHelper.ffill_transition)
        # team_poss = team_poss.reset_index(level=0, drop=True)
        # self.traces["team_poss"] = team_poss.fillna(method="bfill").fillna(method="ffill")

    def estimate_naive_team_poss(self):
        xy_cols = [f"{p}{t}" for p in self.team1_players + self.team2_players for t in ["_x", "_y"]]
        team_poss = pd.Series(index=self.traces.index, dtype=str)

        for phase in self.traces["phase"].unique():
            if type(phase) == str:  # For GPS-event traces, ignore phases with n_players < 22
                phase_tuple = [int(i) for i in phase[1:-1].split(",")]
                if phase_tuple[0] < 0 or phase_tuple[1] < 0:
                    continue

            phase_traces = self.traces[self.traces["phase"] == phase]
            phase_gks = SoccerDataset.detect_goalkeepers(phase_traces)
            team1_code, team2_code = phase_gks[0][0], phase_gks[1][0]

            ball_in_left = phase_traces[xy_cols].mean(axis=1) < self.pitch_size[0] / 2
            team_poss.loc[phase_traces.index] = np.where(ball_in_left, team1_code, team2_code)

        return team_poss

    @staticmethod
    def predict_episode(
        model: nn.Module,
        input: torch.Tensor,
        macro_target: torch.Tensor = None,
        micro_target: torch.Tensor = None,
        masking_prob=1,
        split=False,
        vary_weights=True,
        wlen=100,
    ) -> torch.Tensor:
        device = next(model.parameters()).device.type
        input = input.unsqueeze(0).to(device)

        if not split:
            if isinstance(model, PlayerBall):
                macro_target = macro_target.unsqueeze(0).to(device) if macro_target is not None else None
                micro_target = micro_target.unsqueeze(0).to(device) if micro_target is not None else None
                return model.forward(input, macro_target, micro_target, masking_prob).squeeze(0).detach().cpu()
            elif isinstance(model, PIVRNN):
                return model.sample(input).squeeze(0).detach().cpu()
            else:  # Non-hierarchical LSTM models
                return model.forward(input).squeeze(0).detach().cpu()

        else:
            if model.target_type == "player_poss":
                output_dim = input.shape[-1] // model.params["n_features"]  # 26 in general
            elif model.target_type == "gk":
                output_dim = 4
            else:  # if model.target_type in ["team_poss", "ball"]
                output_dim = 2

            hierarchical = model.model_type.startswith("macro")
            if hierarchical:
                if model.macro_type == "team_poss":
                    output_dim += 2
                elif model.macro_type == "player_poss":
                    output_dim += input.shape[-1] // model.params["n_features"]  # 26 in general

            episode_pred = torch.zeros(input.shape[1], output_dim)
            weights = torch.linspace(1 / wlen, 1 - 1 / wlen, wlen // 2).unsqueeze(1) if vary_weights else 0.5

            for i in range(input.shape[1] // (wlen // 2)):
                i_from = (wlen // 2) * i
                i_half = (wlen // 2) * (i + 1)
                i_to = (wlen // 2) * (i + 2)

                window_pred = model.forward(input[:, i_from:i_to]).squeeze(0).detach().cpu()

                if i == 0:
                    episode_pred[i_from:i_half] += window_pred[: (wlen // 2)]
                else:
                    episode_pred[i_from:i_half] += window_pred[: (wlen // 2)] * weights

                if i == input.shape[1] // (wlen // 2) - 1:
                    episode_pred[i_half:i_to] += window_pred[(wlen // 2) :]
                else:
                    episode_pred[i_half:i_to] += window_pred[(wlen // 2) :] * (1 - weights)

            return episode_pred

    def predict(self, model: nn.Module, masking_prob=1, split=False, evaluate=True):
        target_type = model.target_type
        macro_type = model.macro_type if model.model_type.startswith("macro") else None

        n_features = model.x_dim if target_type == "player_poss" else model.params["n_features"]
        n_input_players = 10 if target_type == "gk" else 11
        feature_types = TraceHelper.player_to_cols("")[:n_features]
        player_cols = [f"{p}{t}" for p in self.team1_players + self.team2_players for t in feature_types]

        if macro_type == "player_poss" or target_type == "player_poss":
            outside_labels = ["OUT-L", "OUT-R", "OUT-B", "OUT-T"]
            outside_x = [0, self.pitch_size[0], self.pitch_size[0] / 2, self.pitch_size[0] / 2]
            outside_y = [self.pitch_size[1] / 2, self.pitch_size[1] / 2, 0, self.pitch_size[1]]

            for i, label in enumerate(outside_labels):
                self.traces[f"{label}_x"] = float(outside_x[i])
                self.traces[f"{label}_y"] = float(outside_y[i])
                self.traces[[f"{label}_vx", f"{label}_vy", f"{label}_speed", f"{label}_accel"]] = 0

            poss_labels = self.team1_players + self.team2_players + outside_labels
            player_cols = [f"{p}{t}" for p in poss_labels for t in feature_types]

            if macro_type == "player_poss":
                macro_pred_df = pd.DataFrame(index=self.traces.index, columns=poss_labels, dtype=float)
            else:  # micro_type == "player_poss"
                micro_pred_df = pd.DataFrame(index=self.traces.index, columns=poss_labels, dtype=float)

        if target_type == "gk":
            gks = []
            for phase in self.traces["phase"].unique():
                phase_traces = self.traces[self.traces["phase"] == phase]
                phase_gks = SoccerDataset.detect_goalkeepers(phase_traces)
                for gk in phase_gks:
                    if gk not in gks:
                        gks.append(gk)

            gks.sort()
            pred_cols = [f"{p}{t}" for p in gks for t in feature_types[:2]]
            micro_pred_df = pd.DataFrame(index=self.traces.index, columns=pred_cols, dtype=float)

        elif target_type == "ball":
            micro_pred_df = pd.DataFrame(index=self.traces.index, columns=["ball_x", "ball_y"], dtype=float)

        elif target_type == "transition":
            micro_pred_df = pd.DataFrame(index=self.traces.index, columns=["transition"], dtype=float)

        if macro_type is None:
            macro_pred_df = None
        else:
            macro_cols = ["A", "B"] if macro_type == "team_poss" else poss_labels
            macro_pred_df = pd.DataFrame(index=self.traces.index, columns=macro_cols, dtype=float)

        n_frames = 0
        if evaluate:
            correct_team_poss = 0
            correct_player_poss = 0
            sum_pos_error = 0
            sum_real_loss = 0

        for phase in self.traces["phase"].unique():
            if type(phase) == str:  # For GPS-event traces, ignore phases with n_players < 22
                phase_tuple = [int(i) for i in phase[1:-1].split(",")]
                if phase_tuple[0] < 0 or phase_tuple[1] < 0:
                    continue

            phase_traces = self.traces[self.traces["phase"] == phase]
            phase_gks = SoccerDataset.detect_goalkeepers(phase_traces)
            team1_code, team2_code = phase_gks[0][0], phase_gks[1][0]

            if target_type == "gk":
                input_cols = [c for c in phase_traces[player_cols].dropna(axis=1).columns if c[:3] not in phase_gks]
                output_cols = np.array([[f"{p}_x", f"{p}_y"] for p in phase_gks]).flatten().tolist()
            else:
                input_cols = [c for c in phase_traces[player_cols].dropna(axis=1).columns]
                if target_type == "ball":
                    output_cols = ["ball_x", "ball_y"]

            team1_cols = [c for c in input_cols if c.startswith(team1_code)]
            team2_cols = [c for c in input_cols if c.startswith(team2_code)]
            input_cols = team1_cols + team2_cols  # Reorder teams so that the left team comes first

            if macro_type == "player_poss" or target_type == "player_poss":
                input_cols += [f"{label}{t}" for label in outside_labels for t in feature_types]
                team_poss_dict = {team1_code: 0, team2_code: 1, "O": 2, "G": 2}

                if macro_type == "player_poss":
                    macro_cols = [c.split("_")[0] for c in input_cols[::n_features]]
                    player_poss_dict = dict(zip(macro_cols, np.arange(len(macro_cols))))
                    player_poss_dict["GOAL-L"] = len(macro_cols) - 4  # same as OUT-L
                    player_poss_dict["GOAL-R"] = len(macro_cols) - 3  # same as OUT-R

                if target_type == "player_poss":
                    output_cols = [c.split("_")[0] for c in input_cols[::n_features]]
                    player_poss_dict = dict(zip(output_cols, np.arange(len(output_cols))))
                    player_poss_dict["GOAL-L"] = len(output_cols) - 4  # same as OUT-L
                    player_poss_dict["GOAL-R"] = len(output_cols) - 3  # same as OUT-R

            if min(len(team1_cols), len(team2_cols)) < n_features * n_input_players:
                continue

            episodes = [e for e in phase_traces["episode"].unique() if e > 0]
            for episode in tqdm(episodes, desc=f"Phase {phase}"):
                episode_traces = phase_traces[phase_traces["episode"] == episode]
                episode_input = torch.FloatTensor(episode_traces[input_cols].values)

                macro_target = None
                if macro_type == "team_poss":
                    macro_target = torch.LongTensor((episode_traces["team_poss"] == team2_code).values)
                elif macro_type == "player_poss":
                    player_poss = episode_traces["player_poss"].fillna(method="bfill").fillna(method="ffill")
                    macro_target = torch.LongTensor(player_poss.map(player_poss_dict).values)

                micro_target = None
                if target_type == "player_poss":
                    player_poss = episode_traces["player_poss"].fillna(method="bfill").fillna(method="ffill")
                    micro_target = torch.LongTensor(player_poss.map(player_poss_dict).values)
                elif target_type in ["gk", "ball"]:
                    micro_target = torch.FloatTensor(episode_traces[output_cols].values)

                with torch.no_grad():
                    try:
                        pred = TraceHelper.predict_episode(
                            model, episode_input, macro_target, micro_target, masking_prob, split
                        )  # [10:-10]
                    except RuntimeError:
                        return episode_input, macro_target, micro_target

                if macro_type is None:
                    micro_pred = pred
                else:
                    macro_pred = pred[:, :-2]
                    micro_pred = pred[:, -2:]
                    macro_pred_probs = nn.Softmax(dim=-1)(macro_pred).numpy()
                    if macro_type == "team_poss":
                        macro_pred_df.loc[episode_traces.index, team1_code] = macro_pred_probs[:, 0]
                        macro_pred_df.loc[episode_traces.index, team2_code] = macro_pred_probs[:, 1]
                    elif macro_type == "player_poss":
                        macro_pred_df.loc[episode_traces.index, macro_cols] = macro_pred_probs

                if target_type in ["team_poss", "player_poss", "transition"]:
                    micro_pred = nn.Softmax(dim=-1)(micro_pred)

                if target_type == "transition":
                    micro_pred_df.loc[episode_traces.index, "transition"] = micro_pred[:, 1].numpy()
                else:
                    micro_pred_df.loc[episode_traces.index, output_cols] = micro_pred.numpy()

                n_frames += micro_pred.shape[0]
                if evaluate:
                    # macro_target = macro_target[10:-10]
                    if macro_type == "team_poss":
                        correct_team_poss += ((macro_pred_probs[:, 1] > 0.5) == macro_target).astype(int).sum()

                    elif macro_type == "player_poss":
                        team_poss_pred = np.argmax(macro_pred.numpy(), axis=1) // 11
                        team_poss_target = player_poss.apply(lambda x: x[0]).map(team_poss_dict)
                        correct_team_poss += (team_poss_pred == team_poss_target).sum()
                        correct_player_poss += calc_class_acc(macro_pred, macro_target, aggfunc="sum")

                    assert micro_target is not None
                    # micro_target = micro_target[10:-10]

                    if target_type == "player_poss":
                        team_poss_pred = np.argmax(micro_pred.numpy(), axis=1) // 11
                        team_poss_target = player_poss.apply(lambda x: x[0]).map(team_poss_dict)
                        correct_team_poss += (team_poss_pred == team_poss_target).sum()
                        correct_player_poss += calc_class_acc(micro_pred, micro_target, aggfunc="sum")

                    elif target_type == "gk":
                        team1_gk_pred = micro_pred[:, 0:2]
                        team2_gk_pred = micro_pred[:, 2:4]
                        team1_gk_target = micro_target[:, 0:2]
                        team2_gk_target = micro_target[:, 2:4]
                        sum_pos_error += calc_trace_dist(team1_gk_pred, team1_gk_target, aggfunc="sum")
                        sum_pos_error += calc_trace_dist(team2_gk_pred, team2_gk_target, aggfunc="sum")

                    elif target_type == "ball":
                        sum_pos_error += calc_trace_dist(micro_pred, micro_target, aggfunc="sum")
                        sum_real_loss += calc_real_loss(micro_pred, episode_input, aggfunc="sum").item()

            if macro_type is not None:
                phase_macro_pred_df = macro_pred_df.loc[phase_traces.index]
                macro_pred_df.loc[phase_traces.index] = phase_macro_pred_df.interpolate(limit_direction="both")

            phase_micro_pred_df = micro_pred_df.loc[phase_traces.index]
            micro_pred_df.loc[phase_traces.index] = phase_micro_pred_df.interpolate(limit_direction="both")

        argmax_idxs = np.argpartition(-macro_pred_df.values, range(3), axis=1)[:, :3]
        player_poss_top3 = pd.DataFrame(np.array(macro_pred_df.columns)[argmax_idxs])
        self.traces["pred_poss"] = macro_pred_df.idxmax(axis=1)
        self.traces["pred_poss_top3"] = player_poss_top3.apply(lambda x: x.tolist(), axis=1)
        self.traces["pred_ball_x"] = micro_pred_df["ball_x"]
        self.traces["pred_ball_y"] = micro_pred_df["ball_y"]

        stats = {"n_frames": n_frames}
        if n_frames == 0:
            return None, None, stats

        if evaluate:
            if correct_team_poss > 0:
                stats["correct_team_poss"] = correct_team_poss
                print(f"team_poss_acc: {round(correct_team_poss / n_frames, 4)}")

            if correct_player_poss > 0:
                stats["correct_player_poss"] = correct_player_poss
                print(f"player_poss_acc: {round(correct_player_poss / n_frames, 4)}")

            if target_type == "gk":
                stats["sum_pos_error"] = sum_pos_error / 2
                print(f"pos_error: {round(sum_pos_error / n_frames / 2, 4)}")

            elif target_type == "ball":
                stats["sum_pos_error"] = sum_pos_error
                stats["sum_real_loss"] = sum_real_loss
                print(f"pos_error: {round(sum_pos_error / n_frames, 4)}")
                print(f"real_loss: {round(sum_real_loss / n_frames, 4)}")

        return macro_pred_df, micro_pred_df, stats

    @staticmethod
    def plot_speed_and_accel_curves(traces: pd.DataFrame, players: list = None) -> animation.FuncAnimation:
        FRAME_DUR = 30
        MAX_SPEED = 40
        MAX_ACCEL = 6

        if players is None:
            players = [c[:3] for c in traces.columns if c.endswith("_speed")]
        else:
            players = [p for p in players if f"{p}_speed" in traces.columns]
        players.sort()

        if len(players) > 20:
            print("Error: No more than 20 players")
            return

        fig, axes = plt.subplots(2, 1)
        fig.set_facecolor("w")
        fig.set_size_inches(15, 10)
        plt.rcParams.update({"font.size": 15})

        times = traces["time"].values
        t0 = int(times[0] - 0.1)

        axes[0].set(xlim=(t0, t0 + FRAME_DUR), ylim=(0, MAX_SPEED))
        axes[1].set(xlim=(t0, t0 + FRAME_DUR), ylim=(-MAX_ACCEL, MAX_ACCEL))
        axes[0].set_ylabel("speed")
        axes[1].set_ylabel("aceel")

        for ax in axes:
            ax.grid()
            ax.set_xlabel("time")
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        speed_plots = dict()
        accel_plots = dict()
        colors = [cm.to_hex(plt.cm.tab20(i)) for i in range(20)]

        for i, p in enumerate(players):
            speeds = traces[f"{p}_speed"].values * 3.6
            accels = traces[f"{p}_accel"].values
            (speed_plots[p],) = axes[0].plot(times, speeds, color=colors[i], label=p)
            (accel_plots[p],) = axes[1].plot(times, accels, color=colors[i], label=p)

        axes[0].legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))

        def animate(i):
            for ax in axes:
                ax.set_xlim(10 * i, 10 * i + FRAME_DUR)

        frames = (len(traces) - 10 * FRAME_DUR) // 100 + 1
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200)
        plt.close(fig)

        return anim


if __name__ == "__main__":
    import json

    from models import load_model

    trial = 810
    device = "cuda:0"
    save_path = f"saved/{trial:03d}"
    with open(f"{save_path}/params.json", "r") as f:
        params = json.load(f)
    model = load_model(params["model"], params).to(device)

    model_path = f"saved/{trial}"
    state_dict = torch.load(
        f"{model_path}/model/{params['model']}_state_dict_best.pt",
        map_location=lambda storage, _: storage,
    )
    model.load_state_dict(state_dict)

    match_id = "20862-20875"
    match_traces = pd.read_csv(f"data/gps_event_traces_gk_pred/{match_id}.csv", header=0, encoding="utf-8-sig")
    helper = TraceHelper(match_traces)
    pred_poss = helper.predict(model, split=False, evaluate=True)
