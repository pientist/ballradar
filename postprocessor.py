import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from matplotlib import animation
from tqdm import tqdm


class Postprocessor:
    def __init__(self, traces: pd.DataFrame, pred_poss: pd.DataFrame, pred_traces: pd.DataFrame):
        self.pred_poss = pred_poss.dropna(axis=0, how="all").copy()
        self.pred_traces = pred_traces.loc[self.pred_poss.index, ["ball_x", "ball_y"]].copy()
        self.traces = traces.loc[self.pred_poss.index].copy()

        self.pred_traces["time"] = self.traces["time"]
        self.target_traces = self.traces[["ball_x", "ball_y"]] if "ball_x" in traces.columns else None

        self.poss_scores = pd.DataFrame(index=self.traces.index, columns=pred_poss.columns, dtype=float)

        output_cols = ["carrier", "ball_x", "ball_y", "focus_x", "focus_y"]
        self.output = pd.DataFrame(index=self.traces.index, columns=output_cols)
        self.output[output_cols[1:]] = self.output[output_cols[1:]].astype(float)

    @staticmethod
    def calc_ball_features(ball_traces: pd.DataFrame) -> pd.DataFrame:
        W_LEN = 7
        P_ORDER = 2

        ball_traces = ball_traces.dropna(subset=["ball_x"])
        times = ball_traces["time"].values

        x = ball_traces["ball_x"].values
        y = ball_traces["ball_y"].values
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
    def calc_ball_dists(player_traces: pd.DataFrame, ball_traces: pd.DataFrame, players: list) -> pd.DataFrame:
        # Calculate distances from the ball to the players
        player_xy_cols = [f"{p}{t}" for p in players for t in ["_x", "_y"]]
        player_xy = player_traces[player_xy_cols].values.reshape(player_traces.shape[0], -1, 2)
        pred_xy = ball_traces[["ball_x", "ball_y"]].values[:, np.newaxis, :]
        ball_dists = np.linalg.norm(pred_xy - player_xy, axis=-1)
        ball_dists = pd.DataFrame(ball_dists, index=player_traces.index, columns=players)

        # Calculate distances from the ball to the pitch lines
        ball_dists["OUT-L"] = (player_traces["OUT-L_x"] - ball_traces["ball_x"]).abs()
        ball_dists["OUT-R"] = (player_traces["OUT-R_x"] - ball_traces["ball_x"]).abs()
        ball_dists["OUT-B"] = (player_traces["OUT-B_y"] - ball_traces["ball_y"]).abs()
        ball_dists["OUT-T"] = (player_traces["OUT-T_y"] - ball_traces["ball_y"]).abs()

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
        pred_traces: pd.DataFrame,
        players: list,
        max_accel=5,
    ) -> tuple[pd.Series, pd.DataFrame]:
        accels = pred_traces[["accel"]].copy()
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

        ball_dists = Postprocessor.calc_ball_dists(traces, pred_traces, players)

        for i in range(len(max_idxs_grouped)):
            start_idx = pred_traces.loc[min_idxs_grouped[i], "accel"].idxmin()
            end_idx = pred_traces.loc[max_idxs_grouped[i], "accel"].idxmax()
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
        pred_poss: pd.DataFrame,
        pred_traces: pd.DataFrame,
        players: list,
        thres_touch=0.2,
        thres_carry=0.5,
    ) -> pd.DataFrame:
        ball_dists = Postprocessor.calc_ball_dists(traces, pred_traces, players)

        poss_scores = pred_poss[players] / np.sqrt(ball_dists + 1e-6)
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
    def finetune_ball_trace(
        player_traces: pd.DataFrame,
        ball_trace: pd.DataFrame,
        carry_records: pd.DataFrame = None,
    ) -> pd.DataFrame:
        output_cols = ["carrier", "ball_x", "ball_y", "focus_x", "focus_y"]
        output = pd.DataFrame(index=player_traces.index, columns=output_cols)
        output[output_cols[1:]] = output[output_cols[1:]].astype(float)

        # Reconstruct the ball trace
        for i in carry_records.index:
            start_idx = carry_records.at[i, "start_idx"]
            end_idx = carry_records.at[i, "end_idx"]
            carrier = carry_records.at[i, "carrier"]

            output.loc[start_idx:end_idx, "carrier"] = carrier
            if not carrier.startswith("OUT"):
                output.loc[start_idx:end_idx, "ball_x"] = player_traces.loc[start_idx:end_idx, f"{carrier}_x"]
                output.loc[start_idx:end_idx, "ball_y"] = player_traces.loc[start_idx:end_idx, f"{carrier}_y"]
            elif carrier in ["OUT-L", "OUT-R"]:
                output.loc[start_idx:end_idx, "ball_x"] = player_traces[f"{carrier}_x"].iloc[0]
                output.loc[start_idx:end_idx, "ball_y"] = ball_trace.loc[start_idx:end_idx, "ball_y"].mean()
            else:  # carrier in ["OUT-B", "OUT-T"]
                output.loc[start_idx:end_idx, "ball_x"] = ball_trace.loc[start_idx:end_idx, "ball_x"].mean()
                output.loc[start_idx:end_idx, "ball_y"] = player_traces[f"{carrier}_y"].iloc[0]

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
                output.at[start_idx, "focus_x"] = player_traces.at[start_idx, f"{carrier}_x"]
                output.at[start_idx, "focus_y"] = player_traces.at[start_idx, f"{carrier}_y"]
                output.at[end_idx, "focus_x"] = player_traces.at[end_idx, f"{carrier}_x"]
                output.at[end_idx, "focus_y"] = player_traces.at[end_idx, f"{carrier}_y"]

        output[["focus_x", "focus_y"]] = output[["focus_x", "focus_y"]].interpolate(limit_direction="both")
        output["focus_x"] = output["focus_x"].clip(0, 108)
        output["focus_y"] = output["focus_y"].clip(18, 54)

        return output

    def run(self, method="ball_accel", max_accel=5, thres_touch=0.2, thres_carry=0.5, evaluate=False):
        if evaluate:
            n_frames = 0
            sum_macro_acc = 0
            sum_micro_pos_errors = 0

        for phase in tqdm(self.traces["phase"].unique()):
            phase_traces = self.traces.loc[self.traces["phase"] == phase].copy()
            phase_pred_poss = self.pred_poss.loc[phase_traces.index]
            phase_pred_traces = self.pred_traces.loc[phase_traces.index]

            players = phase_pred_poss.dropna(axis=1).columns
            for p in players:
                phase_pred_poss[p] = signal.savgol_filter(phase_pred_poss[p], window_length=11, polyorder=2)
            self.pred_poss.loc[phase_traces.index] = phase_pred_poss

            if method == "ball_accel":
                episodes = [e for e in phase_traces["episode"].unique() if e > 0]

                for episode in tqdm(episodes, desc=f"Phase {phase}"):
                    ep_traces = self.traces[self.traces["episode"] == episode].copy()
                    ep_pred_traces = phase_pred_traces.loc[ep_traces.index]

                    ep_pred_traces = Postprocessor.calc_ball_features(ep_pred_traces)
                    carry_records = Postprocessor.detect_carries_by_accel(ep_traces, ep_pred_traces, players, max_accel)

                    ep_output = Postprocessor.finetune_ball_trace(ep_traces, ep_pred_traces, carry_records)
                    self.output.loc[ep_traces.index] = ep_output

                    if evaluate:
                        n_frames += ep_traces.shape[0]

                        ep_target_poss = ep_traces["event_player"].fillna(method="bfill").fillna(method="ffill")
                        ep_output_poss = ep_output["carrier"].fillna(method="bfill").fillna(method="ffill")
                        sum_macro_acc += (ep_target_poss == ep_output_poss).astype(int).sum()

                        ep_target_traces = self.target_traces.loc[ep_traces.index]
                        error_x = (ep_output["ball_x"] - ep_target_traces["ball_x"]).values
                        error_y = (ep_output["ball_y"] - ep_target_traces["ball_y"]).values
                        sum_micro_pos_errors += np.sqrt((error_x**2 + error_y**2).astype(float)).sum()

                self.output.loc[phase_traces.index, ["ball_x", "ball_y", "focus_x", "focus_y"]] = self.output.loc[
                    phase_traces.index, ["ball_x", "ball_y", "focus_x", "focus_y"]
                ].interpolate(limit_direction="both")

            elif method == "poss_score":
                carry_records, poss_scores = Postprocessor.detect_carries_by_poss_score(
                    phase_traces, phase_pred_poss, phase_pred_traces, players, thres_touch, thres_carry
                )
                phase_output = Postprocessor.finetune_ball_trace(phase_traces, phase_pred_traces, carry_records)
                self.output.loc[phase_traces.index] = phase_output

                cols = [c for c in poss_scores.columns if not c.startswith("max")]
                self.poss_scores.loc[phase_traces.index, cols] = poss_scores[cols].values

                if evaluate:
                    episodes = [e for e in phase_traces["episode"].unique() if e > 0]

                    for episode in episodes:
                        ep_traces = self.traces[self.traces["episode"] == episode].copy()
                        ep_output = self.output.loc[ep_traces.index]
                        n_frames += ep_traces.shape[0]

                        ep_target_poss = ep_traces["event_player"].fillna(method="bfill").fillna(method="ffill")
                        ep_output_poss = ep_output["carrier"].fillna(method="bfill").fillna(method="ffill")
                        sum_macro_acc += (ep_target_poss == ep_output_poss).astype(int).sum()

                        ep_target_traces = self.target_traces.loc[ep_traces.index]
                        error_x = (ep_output["ball_x"] - ep_target_traces["ball_x"]).values
                        error_y = (ep_output["ball_y"] - ep_target_traces["ball_y"]).values
                        sum_micro_pos_errors += np.sqrt((error_x**2 + error_y**2).astype(float)).sum()

        if evaluate and n_frames > 0:
            print(f"macro_acc: {round(sum_macro_acc / n_frames, 4)}")
            print(f"micro_pos_error: {round(sum_micro_pos_errors / n_frames, 4)}")

    @staticmethod
    def plot_speeds_and_accels(times: pd.Series, ball_traces: pd.DataFrame, carry_records: pd.DataFrame):
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
    def plot_poss_values(
        times: pd.Series,
        poss_scores: pd.DataFrame,
        valid_players=None,
        valid_score=0,
        carry_records=None,
        ylabel="Possession score",
        save_filename=None,
    ):
        plt.rcParams.update({"font.size": 18})
        plt.figure(figsize=(15, 5))

        if valid_players is None:
            valid_players = poss_scores.max()[poss_scores.max() > valid_score].index

        cmap = "tab10" if len(valid_players) <= 10 else "tab20"
        color_dict = dict(zip(valid_players, plt.cm.get_cmap(cmap).colors))

        # times = poss_scores.index.values / 10
        for p in valid_players:
            plt.plot(times, poss_scores[p], label=p, color=color_dict[p])

        if carry_records is not None:
            for i in carry_records.index:
                start_time = carry_records.loc[i, "start_idx"] / 10
                end_time = carry_records.loc[i, "end_idx"] / 10

                if "carrier" in carry_records.columns:
                    carrier = carry_records.loc[i, "carrier"]
                    color = color_dict[carrier]
                else:
                    color = "gray"

                plt.axvspan(start_time, end_time, alpha=0.5, color=color)

        xmin = (times.iloc[0] // 5) * 5
        plt.xlim(xmin, xmin + 40)
        plt.ylim(0, 1.05)
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

        if save_filename is not None:
            plt.savefig(f"img/{save_filename}.png", bbox_inches="tight")
