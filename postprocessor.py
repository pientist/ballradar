import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from matplotlib import animation
from tqdm import tqdm


class Postprocessor:
    def __init__(self, traces: pd.DataFrame, poss_probs: pd.DataFrame):
        self.poss_probs = poss_probs.dropna(axis=0, how="all").copy()
        self.traces = traces.loc[self.poss_probs.index].copy()

        self.poss_scores = pd.DataFrame(index=self.traces.index, columns=poss_probs.columns, dtype=float)
        self.carry_records = None

        output_cols = ["carrier", "player_poss", "ball_x", "ball_y", "focus_x", "focus_y"]
        self.output = pd.DataFrame(index=self.traces.index, columns=output_cols)
        self.output[output_cols[1:]] = self.output[output_cols[1:]].astype(float)

    @staticmethod
    def calc_ball_features(ball_traces: pd.DataFrame) -> pd.DataFrame:
        W_LEN = 7
        P_ORDER = 2

        ball_traces = ball_traces.dropna(subset=["pred_ball_x"])
        times = ball_traces["time"].values

        x = ball_traces["pred_ball_x"].values
        y = ball_traces["pred_ball_y"].values
        x = signal.savgol_filter(x, window_length=W_LEN, polyorder=P_ORDER)
        y = signal.savgol_filter(y, window_length=W_LEN, polyorder=P_ORDER)

        vx = np.diff(x, prepend=x[0]) / 0.1
        vy = np.diff(y, prepend=y[0]) / 0.1
        vx = signal.savgol_filter(vx, window_length=W_LEN, polyorder=P_ORDER)
        vy = signal.savgol_filter(vy, window_length=W_LEN, polyorder=P_ORDER)
        speeds = np.sqrt(vx**2 + vy**2)

        accels = np.diff(speeds, prepend=speeds[-1]) / 0.1
        accels[:2] = 0
        accels[-2:] = 0
        accels = signal.savgol_filter(accels, window_length=W_LEN, polyorder=P_ORDER)

        cols = ["time", "ball_x", "ball_y", "vx", "vy", "speed", "accel"]
        ball_traces_arr = np.stack((times, x, y, vx, vy, speeds, accels), axis=1)
        return pd.DataFrame(ball_traces_arr, index=ball_traces.index, columns=cols)

    @staticmethod
    def calc_ball_dists(traces: pd.DataFrame, players: list) -> pd.DataFrame:
        # Calculate distances from the ball to the players
        player_xy_cols = [f"{p}{t}" for p in players for t in ["_x", "_y"]]
        player_xy = traces[player_xy_cols].values.reshape(traces.shape[0], -1, 2)
        pred_xy = traces[["pred_ball_x", "pred_ball_y"]].values[:, np.newaxis, :]
        ball_dists = np.linalg.norm(pred_xy - player_xy, axis=-1)
        ball_dists = pd.DataFrame(ball_dists, index=traces.index, columns=players)

        # Calculate distances from the ball to the pitch lines
        ball_dists["OUT-L"] = (traces["OUT-L_x"] - traces["pred_ball_x"]).abs()
        ball_dists["OUT-R"] = (traces["OUT-R_x"] - traces["pred_ball_x"]).abs()
        ball_dists["OUT-B"] = (traces["OUT-B_y"] - traces["pred_ball_y"]).abs()
        ball_dists["OUT-T"] = (traces["OUT-T_y"] - traces["pred_ball_y"]).abs()

        return ball_dists

    @staticmethod
    def generate_carry_records(carriers: pd.Series):
        carriers_prev = carriers.fillna(method="ffill")
        carriers_next = carriers.fillna(method="bfill")
        carriers = carriers_prev.where(carriers_prev == carriers_next, np.nan)

        poss_changes = carriers.notna().astype(int).diff().fillna(0)
        start_idxs = poss_changes[poss_changes > 0].index.values.tolist()
        end_idxs = poss_changes[poss_changes < 0].index.values.tolist()
        if not start_idxs:
            start_idxs = [carriers.index[0]]
        if not end_idxs:
            end_idxs = [carriers.index[-1]]

        if start_idxs[0] > end_idxs[0]:
            start_idxs.insert(0, carriers.index[0])
        if start_idxs[-1] > end_idxs[-1]:
            end_idxs.append(carriers.index[-1])

        carry_records = pd.DataFrame(np.stack([start_idxs, end_idxs], axis=1), columns=["start_idx", "end_idx"])
        carry_records["carrier"] = carriers.loc[start_idxs].values.tolist()

        return carry_records

    @staticmethod
    def detect_carries_by_accel(
        traces: pd.DataFrame,
        ball_features: pd.DataFrame,
        players: list,
        max_accel=5,
    ) -> tuple[pd.Series, pd.DataFrame]:
        accels = ball_features[["accel"]].copy()
        for k in np.arange(2) + 1:
            accels[f"prev{k}"] = accels["accel"].shift(k, fill_value=0)
            accels[f"next{k}"] = accels["accel"].shift(-k, fill_value=0)

        max_flags = (accels["accel"] == accels.max(axis=1)) & (accels["accel"] > max_accel)
        min_flags = (accels["accel"] == accels.min(axis=1)) & (accels["accel"] < -max_accel)
        max_idxs = accels[max_flags].index.tolist()
        min_idxs = accels[min_flags].index.tolist()

        if traces.index[0] in max_idxs:
            max_idxs.pop(0)
        if traces.index[-1] in min_idxs:
            min_idxs.pop(-1)

        if not min_idxs:
            min_idxs.insert(0, traces.index[0])
        if not max_idxs:
            max_idxs.append(traces.index[-1])

        if min_idxs[0] > max_idxs[0]:
            min_idxs.insert(0, traces.index[0])
        if min_idxs[-1] > max_idxs[-1]:
            max_idxs.append(traces.index[-1])

        max_idxs_grouped = []
        min_idxs_grouped = []
        carry_records = []

        while max_idxs:
            # let the local minima belong to the same group if there is no local maximum between them
            min_group = []
            while min_idxs and min_idxs[0] < max_idxs[0]:
                min_group.append(min_idxs.pop(0))
            min_idxs_grouped.append(min_group)

            # let the local maxima belong to the same group if there is no local minimum between them
            max_group = []
            if min_idxs:
                while max_idxs and max_idxs[0] < min_idxs[0]:
                    max_group.append(max_idxs.pop(0))
            else:
                while max_idxs:
                    max_group.append(max_idxs.pop(0))
            max_idxs_grouped.append(max_group)

        ball_dists = Postprocessor.calc_ball_dists(traces, players)

        for i in range(len(max_idxs_grouped)):
            start_idx = ball_features.loc[min_idxs_grouped[i], "accel"].idxmin()
            end_idx = ball_features.loc[max_idxs_grouped[i], "accel"].idxmax()
            carry_record = [start_idx, end_idx]

            if i == 0 or ball_dists.loc[start_idx:end_idx].min().min() < 4:
                carrier = ball_dists.loc[start_idx:end_idx].mean().idxmin()
                traces.loc[start_idx:end_idx, "carrier"] = carrier
                carry_record.append(carrier)

            carry_records.append(carry_record)

        # carry_records = pd.DataFrame(carry_records, columns=["start_idx", "end_idx", "carrier"])
        return Postprocessor.generate_carry_records(traces["carrier"])

    @staticmethod
    def detect_carries_by_poss_score(
        traces: pd.DataFrame,
        poss_probs: pd.DataFrame,
        players: list,
        thres_touch=0.2,
        thres_carry=0.5,
    ) -> pd.DataFrame:
        ball_dists = Postprocessor.calc_ball_dists(traces, players)

        poss_scores = poss_probs[players] / np.sqrt(ball_dists + 1e-6)
        for p in players:
            poss_scores[p] = signal.savgol_filter(poss_scores[p], window_length=11, polyorder=2)

        poss_scores["idxmax"] = poss_scores[players].idxmax(axis=1)
        poss_scores["max"] = poss_scores[players].max(axis=1)

        max_cols = ["max_prev2", "max_prev1", "max", "max_next1", "max_next2"]
        for k in np.arange(2) + 1:
            poss_scores[f"max_prev{k}"] = poss_scores["max"].shift(k, fill_value=0)
            poss_scores[f"max_next{k}"] = poss_scores["max"].shift(-k, fill_value=0)

        at_peak = (poss_scores["max"] > thres_touch) & (poss_scores["max"] == poss_scores[max_cols].max(axis=1))
        in_control = poss_scores["max"] > thres_carry
        poss_scores["carrier"] = np.where(at_peak | in_control, poss_scores["idxmax"], np.nan)

        carry_records = Postprocessor.generate_carry_records(poss_scores["carrier"])
        return carry_records, poss_scores

    @staticmethod
    def finetune_ball_trace(traces: pd.DataFrame, carry_records: pd.DataFrame = None) -> pd.DataFrame:
        output_cols = ["carrier", "ball_x", "ball_y", "focus_x", "focus_y"]
        output = pd.DataFrame(index=traces.index, columns=output_cols)
        output[output_cols[1:]] = output[output_cols[1:]].astype(float)

        # Reconstruct the ball trace
        for i in carry_records.index:
            start_idx = carry_records.at[i, "start_idx"]
            end_idx = carry_records.at[i, "end_idx"]
            carrier = carry_records.at[i, "carrier"]

            output.loc[start_idx:end_idx, "carrier"] = carrier
            if not carrier.startswith("OUT"):
                output.loc[start_idx:end_idx, "ball_x"] = traces.loc[start_idx:end_idx, f"{carrier}_x"]
                output.loc[start_idx:end_idx, "ball_y"] = traces.loc[start_idx:end_idx, f"{carrier}_y"]
            elif carrier in ["OUT-L", "OUT-R"]:
                output.loc[start_idx:end_idx, "ball_x"] = traces[f"{carrier}_x"].iloc[0]
                output.loc[start_idx:end_idx, "ball_y"] = traces.loc[start_idx:end_idx, "pred_ball_y"].mean()
            else:  # carrier in ["OUT-B", "OUT-T"]
                output.loc[start_idx:end_idx, "ball_x"] = traces.loc[start_idx:end_idx, "pred_ball_x"].mean()
                output.loc[start_idx:end_idx, "ball_y"] = traces[f"{carrier}_y"].iloc[0]

        output[["ball_x", "ball_y"]] = output[["ball_x", "ball_y"]].interpolate(limit_direction="both")

        # Calculate xy coordinates to center on when zooming in the panoramic match video
        carry_records["trans_prev"] = 0

        for i in carry_records.index:
            if i == 0:
                send_idx = output.index[0]
            else:
                send_idx = carry_records.at[i - 1, "end_idx"]
            receive_idx = carry_records.at[i, "start_idx"]

            trans_x = abs(output.at[receive_idx, "ball_x"] - output.at[send_idx, "ball_x"])
            trans_y = abs(output.at[receive_idx, "ball_y"] - output.at[send_idx, "ball_y"])
            carry_records.at[i, "trans_prev"] = max(trans_x, trans_y * 0.5)

        carry_records["trans_next"] = carry_records["trans_prev"].shift(-1).fillna(0)
        carry_records["trans_dur"] = carry_records["end_idx"] - carry_records["start_idx"] + 1
        carry_records["focus"] = (carry_records["trans_dur"] > 5) + (
            carry_records[["trans_prev", "trans_next"]].min(axis=1) > 15
        )

        for i in carry_records.index:
            if i > 0 and carry_records.at[i - 1, "focus"] and not carry_records.at[i, "focus"]:
                continue

            else:
                carry_records.at[i, "focus"] = True

                start_idx = carry_records.at[i, "start_idx"]
                end_idx = carry_records.at[i, "end_idx"]
                if i > 0 and i < len(carry_records) - 1:
                    start_idx += min(5, carry_records.at[i, "trans_dur"] - 1)

                carrier = carry_records.at[i, "carrier"]
                output.at[start_idx, "focus_x"] = traces.at[start_idx, f"{carrier}_x"]
                output.at[start_idx, "focus_y"] = traces.at[start_idx, f"{carrier}_y"]
                output.at[end_idx, "focus_x"] = traces.at[end_idx, f"{carrier}_x"]
                output.at[end_idx, "focus_y"] = traces.at[end_idx, f"{carrier}_y"]

        output[["focus_x", "focus_y"]] = output[["focus_x", "focus_y"]].interpolate(limit_direction="both")
        output["focus_x"] = output["focus_x"].clip(0, 108)
        output["focus_y"] = output["focus_y"].clip(18, 54)

        return output

    def run(self, method="ball_accel", max_accel=5, thres_touch=0.2, thres_carry=0.5, evaluate=False):
        if evaluate:
            n_frames = 0
            correct_player_poss = 0
            sum_pos_error = 0

        carry_records_list = []

        for phase in tqdm(self.traces["phase"].unique()):
            phase_traces = self.traces[self.traces["phase"] == phase].copy()
            phase_poss_probs = self.poss_probs.loc[phase_traces.index]

            players = phase_poss_probs.dropna(axis=1).columns
            for p in players:
                phase_poss_probs[p] = signal.savgol_filter(phase_poss_probs[p], window_length=11, polyorder=2)
            self.poss_probs.loc[phase_traces.index] = phase_poss_probs

            if method == "ball_accel":
                episodes = [e for e in phase_traces["episode"].unique() if e > 0]

                for episode in episodes:
                    ep_traces = self.traces[self.traces["episode"] == episode].copy()
                    ball_features = Postprocessor.calc_ball_features(ep_traces)
                    carry_records = Postprocessor.detect_carries_by_accel(ep_traces, ball_features, players, max_accel)
                    ep_output = Postprocessor.finetune_ball_trace(ep_traces, carry_records)

                    carry_records["phase"] = phase
                    carry_records_list.append(carry_records)
                    self.output.loc[ep_traces.index] = ep_output

                    if evaluate:
                        n_frames += ep_traces.shape[0]

                        ep_target_poss = ep_traces["player_poss"].fillna(method="bfill").fillna(method="ffill")
                        ep_output_poss = ep_output["carrier"].fillna(method="bfill").fillna(method="ffill")
                        correct_player_poss += (ep_target_poss == ep_output_poss).astype(int).sum()

                        error_x = (ep_output["ball_x"] - ep_traces["ball_x"]).values
                        error_y = (ep_output["ball_y"] - ep_traces["ball_y"]).values
                        sum_pos_error += np.sqrt((error_x**2 + error_y**2).astype(float)).sum()

                self.output.loc[phase_traces.index, ["ball_x", "ball_y", "focus_x", "focus_y"]] = self.output.loc[
                    phase_traces.index, ["ball_x", "ball_y", "focus_x", "focus_y"]
                ].interpolate(limit_direction="both")

            elif method == "poss_score":
                carry_records, poss_scores = Postprocessor.detect_carries_by_poss_score(
                    phase_traces, phase_poss_probs, players, thres_touch, thres_carry
                )
                phase_output = Postprocessor.finetune_ball_trace(phase_traces, carry_records)

                carry_records["phase"] = phase
                carry_records_list.append(carry_records)
                self.output.loc[phase_traces.index] = phase_output

                cols = [c for c in poss_scores.columns if not c.startswith("max")]
                self.poss_scores.loc[phase_traces.index, cols] = poss_scores[cols].values

                if evaluate:
                    episodes = [e for e in phase_traces["episode"].unique() if e > 0]

                    for episode in episodes:
                        ep_traces = self.traces[self.traces["episode"] == episode].copy()
                        ep_output = self.output.loc[ep_traces.index]
                        n_frames += ep_traces.shape[0]

                        ep_target_poss = ep_traces["player_poss"].fillna(method="bfill").fillna(method="ffill")
                        ep_output_poss = ep_output["carrier"].fillna(method="bfill").fillna(method="ffill")
                        self.output.loc[ep_traces.index, "player_poss"] = ep_output_poss
                        correct_player_poss += (ep_target_poss == ep_output_poss).astype(int).sum()

                        error_x = (ep_output["ball_x"] - ep_traces["ball_x"]).values
                        error_y = (ep_output["ball_y"] - ep_traces["ball_y"]).values
                        sum_pos_error += np.sqrt((error_x**2 + error_y**2).astype(float)).sum()

        self.carry_records = pd.concat(carry_records_list)

        if evaluate and n_frames > 0:
            print(f"player_poss_acc: {round(correct_player_poss / n_frames, 4)}")
            print(f"pos_error: {round(sum_pos_error / n_frames, 4)}")

    @staticmethod
    def plot_speed_and_accel_curves(times: pd.Series, ball_traces: pd.DataFrame, carry_records: pd.DataFrame):
        fig, axes = plt.subplots(2, 1)
        fig.set_facecolor("w")
        fig.set_size_inches(15, 10)
        plt.rcParams.update({"font.size": 15})

        for i in carry_records.index:
            start_time = (carry_records.at[i, "start_idx"] + 1) / 10
            end_time = (carry_records.at[i, "end_idx"] + 1) / 10
            axes[0].axvspan(start_time, end_time, alpha=0.5, color="grey")
            axes[1].axvspan(start_time, end_time, alpha=0.5, color="grey")

        xmin = (times.iloc[0] // 5) * 5
        axes[0].set(xlim=(xmin, xmin + 40), ylim=(0, 20))
        axes[1].set(xlim=(xmin, xmin + 40), ylim=(-20, 20))
        axes[0].plot(times, ball_traces["speed"], color="black")
        axes[1].plot(times, ball_traces["accel"], color="black")

        axes[0].set_ylabel("Speed [m/s]")
        axes[1].set_ylabel("Acceleration [m/s²]")
        axes[1].set_xlabel("Time [s]")

        axes[0].grid()
        axes[1].grid()

    @staticmethod
    def detect_false_poss_segments(pred_poss, true_poss):
        false_idxs = true_poss[true_poss != pred_poss].reset_index()["index"]
        time_diffs = pd.Series(false_idxs.diff().fillna(10).values, index=false_idxs)
        segment_ids = (time_diffs > 3).astype(int).cumsum().rename("segment_id").reset_index()

        start_idxs = segment_ids.groupby("segment_id")["index"].first().rename("start_idx")
        end_idxs = segment_ids.groupby("segment_id")["index"].last().rename("end_idx")
        false_segments = pd.concat([start_idxs, end_idxs], axis=1)

        false_segments["miss"] = False
        false_segments["false_alarm"] = False

        for i in false_segments.index:
            i0 = false_segments.at[i, "start_idx"]
            i1 = false_segments.at[i, "end_idx"]

            true_players = true_poss.loc[i0:i1].unique()
            pred_players = pred_poss.loc[i0:i1].unique()
            true_players_ext = true_poss.loc[i0 - 10 : i1 + 10].unique()
            pred_players_ext = pred_poss.loc[i0 - 10 : i1 + 10].unique()

            false_segments.at[i, "miss"] = len(set(true_players) - set(pred_players_ext)) != 0
            false_segments.at[i, "false_alarm"] = len(set(pred_players) - set(true_players_ext)) != 0

        return false_segments

    @staticmethod
    def plot_poss_and_error_curves(
        traces: pd.DataFrame,
        poss_scores: pd.DataFrame,
        pp_output: pd.DataFrame = None,
    ) -> animation.FuncAnimation:
        FRAME_DUR = 30
        MAX_DIST = 20

        nn_pos_error_xy = traces[["ball_x", "ball_y"]] - traces[["pred_ball_x", "pred_ball_y"]].values
        nn_pos_errors = nn_pos_error_xy.apply(np.linalg.norm, axis=1)
        if pp_output is not None:
            pp_pos_error_xy = traces[["ball_x", "ball_y"]] - pp_output[["ball_x", "ball_y"]].values
            pp_pos_errors = pp_pos_error_xy.apply(np.linalg.norm, axis=1)

        poss_cols = [p for p in poss_scores.dropna(axis=1).columns if p[0] in ["A", "B", "O"]]
        poss_dict = dict(zip(poss_cols, np.arange(len(poss_cols))))
        poss_dict["GOAL-L"] = len(poss_cols) - 4
        poss_dict["GOAL-R"] = len(poss_cols) - 3

        true_poss = traces["player_poss"].fillna(method="bfill").fillna(method="ffill").map(poss_dict)
        nn_pred_poss = traces["pred_poss"].map(poss_dict)
        false_segments = Postprocessor.detect_false_poss_segments(nn_pred_poss, true_poss)
        if pp_output is not None:
            pp_pred_poss = pp_output["player_poss"].fillna(method="bfill").fillna(method="ffill").map(poss_dict)

        fig, axes = plt.subplots(3, 1)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0.05)
        fig.set_size_inches(15, 20)
        plt.rcParams.update({"font.size": 15})

        times = traces["time"].values
        t0 = int(times[0] - 0.1)

        axes[0].plot(times, true_poss, color="tab:blue", marker="o", label="True")
        axes[0].plot(times, nn_pred_poss, color="orangered", marker="o", label="NN output")
        if pp_output is not None:
            axes[0].plot(times, pp_pred_poss, color="darkgreen", marker="o", label="PP output")

        axes[0].set(xlim=(t0, t0 + FRAME_DUR), ylim=(-1, len(poss_cols)))
        axes[0].set_xticklabels([])
        axes[0].set_yticks(ticks=np.arange(len(poss_cols)), labels=poss_cols)
        axes[0].set_ylabel("Ball possessor", fontdict={"size": 20})
        axes[0].grid()
        axes[0].legend(loc="upper right")

        n_players = (len(poss_cols) - 4) // 2
        base_cmaps = ["hot_r", "winter_r", "Greys_r"]
        colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.1, 0.9, n_players)) for name in base_cmaps])
        poss_cols = poss_cols[:n_players] + poss_cols[-4:-2] + poss_cols[n_players:-4] + poss_cols[-2:]
        for p in poss_cols:
            axes[1].plot(times, poss_scores[p], label=p, color=colors[poss_dict[p]])

        axes[1].set(xlim=(t0, t0 + FRAME_DUR), ylim=(0, 1.05))
        axes[1].set_xticklabels([])
        axes[1].set_ylabel("Possession probability", fontdict={"size": 20})
        axes[1].grid(which="major", axis="both")
        axes[1].legend(loc="upper right", ncols=2)

        axes[2].plot(times, nn_pos_errors, color="orangered", label="NN output")
        if pp_output is not None:
            axes[2].plot(times, pp_pos_errors, color="tab:green", label="PP output")
            axes[2].legend(loc="upper right")

        axes[2].set(xlim=(t0, t0 + FRAME_DUR), ylim=(0, MAX_DIST))
        axes[2].set_xlabel("Time [s]", fontdict={"size": 20})
        axes[2].set_ylabel("Position error", fontdict={"size": 20})
        axes[2].grid()

        for i in tqdm(false_segments.index):
            start_time = traces.at[false_segments.at[i, "start_idx"], "time"] - 0.05
            end_time = traces.at[false_segments.at[i, "end_idx"], "time"] + 0.05

            miss = false_segments.at[i, "miss"]
            false_alarm = false_segments.at[i, "false_alarm"]
            if miss and false_alarm:
                color = "red"
            elif miss:
                color = "tab:blue"
            elif false_alarm:
                color = "tab:orange"
            else:
                color = "grey"

            axes[0].axvspan(start_time, end_time, alpha=0.3, color=color)
            axes[1].axvspan(start_time, end_time, alpha=0.3, color=color)
            axes[2].axvspan(start_time, end_time, alpha=0.3, color=color)

        def animate(i):
            for ax in axes:
                ax.set_xlim(10 * i, 10 * i + FRAME_DUR)

        frames = (len(traces) - 10 * FRAME_DUR) // 100 + 1
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=500)
        plt.close(fig)

        return anim
