import math
import os
import sys
from datetime import timedelta

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from tqdm import tqdm

from datatools.trace_helper import TraceHelper


class MetricaHelper(TraceHelper):
    def __init__(
        self,
        team1_traces: pd.DataFrame = None,
        team2_traces: pd.DataFrame = None,
        traces_from_txt: pd.DataFrame = None,
        traces_preprocessed: pd.DataFrame = None,
        events: pd.DataFrame = None,
        pitch_size=(108, 72),
    ):
        if traces_preprocessed is not None:
            traces = traces_preprocessed
            frames_to_episodes = traces[["frame", "episode"]].rename(columns={"frame": "start_frame"})
            events = pd.merge(events, frames_to_episodes)

        else:
            if team1_traces is not None:
                assert team2_traces is not None and traces_from_txt is None
                traces = MetricaHelper.load_traces_from_csv(team1_traces, team2_traces)
            else:
                assert team2_traces is None and traces_from_txt is not None
                traces = traces_from_txt

            x_cols = [c for c in traces.columns if c.endswith("_x")]
            y_cols = [c for c in traces.columns if c.endswith("_y")]
            traces[x_cols] *= pitch_size[0]
            traces[y_cols] *= pitch_size[1]

            players = [c[:-2] for c in traces.columns if c.endswith("_x") and not c.startswith("ball")]
            events = MetricaHelper.load_events(events, players)

        super().__init__(traces, events, pitch_size)

    @staticmethod
    def load_traces_from_csv(team1_traces: pd.DataFrame, team2_traces: pd.DataFrame) -> pd.DataFrame:
        team1_players = [f"A{int(c[2][6:]):02d}" for c in team1_traces.columns[3:-2:2]]
        team1_xy_cols = np.array([[f"{p}_x", f"{p}_y"] for p in team1_players]).flatten().tolist()
        team1_traces.columns = ["session", "frame", "time"] + team1_xy_cols + ["ball_x", "ball_y"]
        team1_traces = team1_traces.set_index("frame").astype(float)
        team1_traces["session"] = team1_traces["session"].astype(int)

        team2_players = [f"B{int(c[2][6:]):02d}" for c in team2_traces.columns[3:-2:2]]
        team2_xy_cols = np.array([[f"{p}_x", f"{p}_y"] for p in team2_players]).flatten().tolist()
        team2_traces.columns = ["session", "frame", "time"] + team2_xy_cols + ["ball_x", "ball_y"]
        team2_traces = team2_traces.set_index("frame").astype(float)
        team2_traces["session"] = team2_traces["session"].astype(int)

        header = team1_traces.columns[:-2].tolist() + team2_traces.columns[2:].tolist()
        traces = pd.merge(team1_traces, team2_traces)[header]
        traces.index = team1_traces.index.astype(int)
        return traces

    @staticmethod
    def load_events(events: pd.DataFrame, players: list) -> pd.DataFrame:
        events.columns = [
            "team",
            "type",
            "subtype",
            "session",
            "start_frame",
            "start_time",
            "end_frame",
            "end_time",
            "from",
            "to",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
        ]

        events.loc[events["subtype"].isna(), "subtype"] = events.loc[events["subtype"].isna(), "type"]
        events["start_time"] = events["start_time"].apply(lambda x: math.ceil(x * 10) / 10).round(1)
        events["end_time"] = events["end_time"].apply(lambda x: math.ceil(x * 10) / 10).round(1)

        player_dict1 = dict(zip([f"Player {int(p[1:])}" for p in players], players))
        player_dict2 = dict(zip([f"Player{int(p[1:])}" for p in players], players))
        player_dict = {**player_dict1, **player_dict2}
        player_dict[np.nan] = np.nan

        events["from"] = events["from"].apply(lambda x: player_dict[x])
        events["to"] = events["to"].apply(lambda x: player_dict[x])

        return events

    def generate_phase_records(self):
        players = self.team1_players + self.team2_players
        player_x_cols = [f"{p}_x" for p in players]

        self.traces["phase"] = 0
        play_records = []
        phase_records = []

        for p in players:
            valid_player_idx = self.traces[self.traces[f"{p}_x"].notna()].index
            f0 = valid_player_idx[0]
            f1 = valid_player_idx[-1]

            if len(self.traces.loc[f1, player_x_cols].dropna()) > 22:
                self.traces.loc[f1, [f"{p}_x", f"{p}_y"]] = np.nan
                play_records.append([p, f0, f1 - 1])
            else:
                play_records.append([p, f0, f1])

        play_records = pd.DataFrame(play_records, columns=["object", "start_frame", "end_frame"]).set_index("object")

        change_frames = play_records["start_frame"].tolist()
        change_frames.extend(
            [
                self.traces[self.traces["session"] == 2].index[0],
                play_records["end_frame"].max(),
            ]
        )
        change_frames = list(set(change_frames))
        change_frames.sort()

        for i, f0 in enumerate(change_frames[:-1]):
            f1 = change_frames[i + 1] - 1
            self.traces.loc[f0:f1, "phase"] = i + 1

            session = self.traces.loc[f0, "session"]
            start_time = round(self.traces.at[f0, "time"], 1)
            end_time = round(self.traces.at[f1 + 1, "time"] - 0.1, 1)

            inplay_flags = self.traces.loc[f0:f1, player_x_cols].notna().any()
            player_codes = [c[:-2] for c in inplay_flags[inplay_flags].index]

            phase_records.append([i + 1, session, start_time, end_time, player_codes])

        header = ["phase", "session", "start_time", "end_time", "player_codes"]
        self.phase_records = pd.DataFrame(phase_records, columns=header).set_index("phase")

    def downsample_to_10fps(self):
        upsample_idxs = pd.date_range("2020-01-01 00:00:00.02", periods=len(self.traces) * 2, freq="0.02S")
        xy_cols = [c for c in self.traces.columns if c.endswith("_x") or c.endswith("_y")]
        traces_50fps = pd.DataFrame(index=upsample_idxs, columns=["session", "phase"] + xy_cols, dtype="float")
        traces_50fps.index.name = "datetime"

        traces_50fps.loc[traces_50fps.index[1::2]] = self.traces[["session", "phase"] + xy_cols].values

        traces_50fps[["session", "phase"]] = traces_50fps[["session", "phase"]].fillna(method="bfill").astype(int)
        traces_50fps = traces_50fps.groupby("session", group_keys=False).apply(
            lambda x: x.interpolate(limit_area="inside")
        )
        traces_50fps = traces_50fps[traces_50fps["phase"] > 0]

        traces_10fps_list = []
        start_dt = pd.to_datetime("2020-01-01 00:00:00")

        for session in traces_50fps["session"].unique():
            session_traces_50fps = traces_50fps[traces_50fps["session"] == session]
            session_traces_10fps = session_traces_50fps.resample("0.1S", closed="right", label="right").mean()

            session_phase_records = self.phase_records[self.phase_records["session"] == session]
            session_start_dt = start_dt + timedelta(seconds=session_phase_records["start_time"].iloc[0])
            session_end_dt = start_dt + timedelta(seconds=session_phase_records["end_time"].iloc[-1])
            traces_10fps_list.append(session_traces_10fps.loc[session_start_dt:session_end_dt])

        traces_10fps = pd.concat(traces_10fps_list).reset_index()
        assert isinstance(traces_10fps, pd.DataFrame)

        traces_10fps[["session", "phase"]] = traces_10fps[["session", "phase"]].astype(int)
        traces_10fps["time"] = (np.arange(len(traces_10fps)) * 0.1 + 0.1).round(1)
        traces_10fps = traces_10fps.set_index("time")

        for i in tqdm(self.events.index, desc="Combining tracking and event data"):
            t0 = self.events.at[i, "start_time"]
            t1 = self.events.at[i, "end_time"]
            traces_10fps.loc[t0:t1, "event_player"] = self.events.at[i, "from"]
            traces_10fps.loc[t0:t1, "event_type"] = self.events.at[i, "subtype"]

        for phase in self.phase_records.index:
            t0 = self.phase_records.at[phase, "start_time"]
            t1 = self.phase_records.at[phase, "end_time"]
            traces_10fps.loc[t0:t1, "phase"] = phase

            valid_players = self.phase_records.at[phase, "player_codes"]
            valid_cols = np.array([[f"{p}_x", f"{p}_y"] for p in valid_players]).flatten().tolist()
            invalid_players = list(set(self.team1_players + self.team2_players) - set(valid_players))
            invalid_cols = np.array([[f"{p}_x", f"{p}_y"] for p in invalid_players]).flatten().tolist()

            traces_10fps.loc[t0:t1, invalid_cols] = np.nan
            traces_10fps_interp = traces_10fps.loc[t0:t1, valid_cols].interpolate(limit_direction="both")
            traces_10fps.loc[t0:t1, valid_cols] = traces_10fps_interp

        traces_10fps["frame"] = np.arange(len(traces_10fps)) + 1
        traces_10fps["episode"] = 0
        traces_10fps["team_poss"] = np.nan
        traces_10fps["player_poss"] = np.nan
        traces_10fps[["ball_x", "ball_y"]] = (
            traces_10fps[["session", "ball_x", "ball_y"]]
            .groupby("session", group_keys=False)
            .apply(lambda x: x.interpolate(limit_direction="both"))[["ball_x", "ball_y"]]
        )

        meta_cols = [
            "frame",
            "session",
            "time",
            "phase",
            "episode",
            "team_poss",
            "player_poss",
            "event_player",
            "event_type",
        ]
        self.traces = traces_10fps.reset_index()[meta_cols + xy_cols]

        self.events["start_frame"] = (self.events["start_time"] * 10).astype(int)
        self.events["end_frame"] = (self.events["end_time"] * 10).astype(int)
        phase_times = self.traces[["time", "phase"]].rename(columns={"time": "start_time"})
        self.events = pd.merge(self.events, phase_times)

    def split_into_episodes(self, margin_sec=2):
        self.traces["episode"] = 0
        count = 0

        for phase in self.traces["phase"].unique():
            phase_traces = self.traces[self.traces["phase"] == phase]
            phase_events = self.events[(self.events["phase"] == phase) & (self.events["type"] != "CARD")]
            assert isinstance(phase_events, pd.DataFrame)

            time_diffs = phase_events["start_time"].diff().fillna(60)
            episodes = (time_diffs > 10).astype(int).cumsum() + count
            count = episodes.max() if not episodes.empty else count

            grouped = phase_events.groupby(episodes)["start_time"]
            first_event_times = grouped.min()
            last_event_times = grouped.max()

            for episode in first_event_times.index:
                first_time = first_event_times.loc[episode] - margin_sec
                last_time = last_event_times.loc[episode] + margin_sec
                episode_idxs = phase_traces[
                    (phase_traces["time"] >= first_time) & (phase_traces["time"] <= last_time)
                ].index
                self.traces.loc[episode_idxs, "episode"] = episode

    def find_gt_player_poss(self):
        self.traces["player_poss"] = np.nan
        if "frame" in self.traces.columns:
            self.traces.set_index("frame", inplace=True)

        events = self.events[
            ~(self.events["type"].isin(["CARRY", "CARD", "SET PIECE"]))
            & ~((self.events["type"] == "BALL LOST") & (self.events["subtype"] == "THEFT"))
            & ~((self.events["type"] == "CHALLENGE") & (self.events["subtype"].str.endswith("-LOST")))
        ].copy()

        type_order = ["BALL LOST", "CHALLENGE", "RECOVERY", "FAULT RECEIVED", "PASS", "SHOT", "BALL OUT"]
        events["type"] = pd.Categorical(events["type"], categories=type_order)
        events.sort_values(["start_time", "type"], inplace=True)

        out_frame = 0

        for i in events.index:
            event_type = events.at[i, "type"]
            event_subtype = events.at[i, "subtype"]
            start_frame = events.at[i, "start_frame"]
            end_frame = events.at[i, "end_frame"]

            if start_frame > out_frame + 20:
                self.traces.at[start_frame, "player_poss"] = events.at[i, "from"]
                if event_type == "PASS":
                    self.traces.at[end_frame, "player_poss"] = events.at[i, "to"]

                if event_type == "BALL OUT" or event_subtype.endswith("-OUT") or event_subtype.endswith("-GOAL"):
                    out_x = events.at[i, "end_x"]
                    out_y = events.at[i, "end_y"]
                    if out_x < 0:
                        out_label = "GOAL-L" if event_subtype.endswith("-GOAL") else "OUT-L"
                    elif out_x > 1:
                        out_label = "GOAL-R" if event_subtype.endswith("-GOAL") else "OUT-R"
                    elif out_y < 0:
                        out_label = "OUT-B"
                    elif out_y > 1:
                        out_label = "OUT-T"
                    else:
                        continue

                    out_frame = end_frame
                    if i == events.index[-1]:
                        self.traces.loc[out_frame:, "player_poss"] = out_label
                    else:
                        i_next = events[events["start_frame"] > out_frame + 20].index[0]
                        next_frame = self.events.at[i_next, "start_frame"]
                        self.traces.loc[out_frame : next_frame - 21, "player_poss"] = out_label
                        self.traces.loc[next_frame - 20 : next_frame, "player_poss"] = self.events.at[i_next, "from"]

        poss_prev = self.traces["player_poss"].fillna(method="ffill")
        poss_next = self.traces["player_poss"].fillna(method="bfill")
        self.traces["player_poss"] = poss_prev.where(poss_prev == poss_next, np.nan)
        self.traces.reset_index(inplace=True)

    @staticmethod
    def find_nearest_player(snapshot, players, team_code=None):
        if team_code is None:
            x_cols = [f"{p}_x" for p in players]
            y_cols = [f"{p}_y" for p in players]
        else:
            x_cols = [f"{p}_x" for p in players if p[0] == team_code]
            y_cols = [f"{p}_y" for p in players if p[0] == team_code]

        ball_dists_x = (snapshot[x_cols] - snapshot["ball_x"]).astype(float).values
        ball_dists_y = (snapshot[y_cols] - snapshot["ball_y"]).astype(float).values

        if team_code is None:
            ball_dists = pd.Series(np.sqrt(ball_dists_x**2 + ball_dists_y**2), index=players)
        else:
            team_players = [p for p in players if p[0] == team_code]
            ball_dists = pd.Series(np.sqrt(ball_dists_x**2 + ball_dists_y**2), index=team_players)

        return ball_dists.idxmin()

    def correct_event_player_ids(self):
        print("\nCorrecting event player IDs:")
        players = self.team1_players + self.team2_players
        player_x_cols = [f"{p}_x" for p in players]
        valid_types = ["BALL LOST", "BALL OUT", "CHALLENGE", "PASS", "RECOVERY", "SET PIECE", "SHOT"]

        for phase in self.events["phase"].unique():
            phase_events = self.events[self.events["phase"] == phase]
            phase_traces = self.traces[self.traces["phase"] == phase]
            phase_players = [c[:-2] for c in phase_traces[player_x_cols].dropna(axis=1).columns]

            switch_counts = pd.DataFrame(0, index=players, columns=players)
            for i in tqdm(phase_events.index, desc=f"Phase {phase}"):
                event_type = phase_events.at[i, "type"]
                event_subtype = phase_events.at[i, "subtype"]

                if event_type in valid_types:
                    if event_type == "BALL LOST" and event_subtype == "THEFT":
                        continue
                    if event_type == "CHALLENGE" and not event_subtype.endswith("-WON"):
                        continue

                    start_frame = phase_events.at[i, "start_frame"]
                    end_frame = phase_events.at[i, "end_frame"]

                    recorded_p_from = phase_events.at[i, "from"]
                    detected_p_from = MetricaHelper.find_nearest_player(
                        phase_traces.loc[start_frame - 1], phase_players, recorded_p_from[0]
                    )
                    switch_counts.at[recorded_p_from, detected_p_from] += 1

                    if event_type == "PASS":
                        recorded_p_to = phase_events.at[i, "to"]
                        detected_p_to = MetricaHelper.find_nearest_player(
                            phase_traces.loc[end_frame - 1], phase_players, recorded_p_to[0]
                        )
                        switch_counts.at[recorded_p_to, detected_p_to] += 1

            switch_dict = switch_counts[switch_counts.sum(axis=1) > 0].idxmax(axis=1).to_dict()
            self.events.loc[phase_events.index, "from"] = phase_events["from"].replace(switch_dict)
            self.events.loc[phase_events.index, "to"] = phase_events["to"].replace(switch_dict)
            self.traces.loc[phase_traces.index, "event_player"] = phase_traces["event_player"].replace(switch_dict)
            self.traces.loc[phase_traces.index, "player_poss"] = phase_traces["player_poss"].replace(switch_dict)

    def generate_pass_records(self, frames: pd.Series = None):
        events = self.events[self.events["start_frame"].isin(frames)] if frames is not None else self.events

        valid_types = ["BALL LOST", "RECOVERY", "PASS"]
        events = events[events["type"].isin(valid_types)].copy()
        events["type"] = pd.Categorical(events["type"], categories=valid_types)
        events.sort_values(["start_time", "type"], ignore_index=True, inplace=True)

        passes = []

        for i in events.index[:-1]:
            event_type = events.at[i, "type"]
            event_subtype = events.at[i, "subtype"]

            if event_type == "PASS" or (event_type == "BALL LOST" and "INTERCEPTION" in event_subtype):
                episode = events.at[i, "episode"]
                start_frame = events.at[i, "start_frame"]
                passer = events.at[i, "from"]

                if event_type == "PASS":
                    end_frame = events.at[i, "end_frame"]
                    receiver = events.at[i, "to"]
                    success = True

                else:  # event_type == BALL LOST and event_subtype contains INTERCEPTION
                    next_events = events[i + 1 : i + 5]
                    recovery = next_events[(next_events["type"] == "RECOVERY") & (next_events["from"] != passer)]
                    if recovery.empty:
                        continue
                    else:
                        i_next = recovery.index[0]
                        end_frame = events.at[i_next, "start_frame"]
                        receiver = events.at[i_next, "from"]
                        success = False

                passes.append([episode, start_frame, end_frame, passer, receiver, success])

        pass_cols = ["episode", "start_frame", "end_frame", "passer", "receiver", "success"]
        return pd.DataFrame(passes, columns=pass_cols)


if __name__ == "__main__":
    match_id = 3

    trace_file = f"data/metrica_traces/Sample_Game_{match_id}/Sample_Game_{match_id}_RawTrackingData.csv"
    events_file = f"data/metrica_traces/Sample_Game_{match_id}/Sample_Game_{match_id}_RawEventsData.csv"
    traces = pd.read_csv(trace_file, index_col=0)
    events = pd.read_csv(events_file, header=0)
    helper = MetricaHelper(traces_from_txt=traces, events=events)

    helper.generate_phase_records()
    helper.downsample_to_10fps()
    # helper.split_into_episodes()
    # helper.calculate_running_feaetures(smoothing=True)
    helper.find_gt_team_poss()
