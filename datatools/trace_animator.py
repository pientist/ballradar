import os
import sys
from typing import Dict

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, axes, collections, lines, text
from matplotlib.patches import Rectangle

import datatools.matplotsoccer as mps


class TraceAnimator:
    def __init__(
        self,
        trace_dict: Dict[str, pd.DataFrame] = None,
        poss_probs: np.ndarray = None,
        show_times=True,
        show_episodes=False,
        show_events=False,
        show_focus=False,
        annot_cols=None,  # column names for additional annotation
        rotate_pitch=False,
        anonymize=False,
        max_frames=np.inf,
        fps=10,
        play_speed=1,
    ):
        self.trace_dict = trace_dict
        self.poss_probs = poss_probs

        self.show_times = show_times
        self.show_episodes = show_episodes
        self.show_events = show_events
        self.show_focus = show_focus
        self.annot_cols = annot_cols
        self.rotate_pitch = rotate_pitch
        self.anonymize = anonymize

        self.max_frames = max_frames
        self.fps = fps
        self.play_speed = play_speed

        anim_cols = ["inplay_records", "team1", "team2", "ball", "focus"]
        self.anim_args = pd.DataFrame(index=self.trace_dict.keys(), columns=anim_cols)

    @staticmethod
    def plot_players(
        traces: pd.DataFrame,
        ax: axes.Axes,
        sizes=750,
        alpha=1,
        anonymize=False,
    ):
        if len(traces.columns) == 0:
            return None

        color = "tab:red" if traces.columns[0].startswith("A") else "tab:blue"
        x = traces[traces.columns[0::2]].values
        y = traces[traces.columns[1::2]].values
        size = sizes[0, 0] if type(sizes) == np.ndarray else sizes
        scat = ax.scatter(x[0], y[0], s=size, c=color, alpha=alpha, zorder=2)

        players = [c[:-2] for c in traces.columns[0::2]]
        player_dict = dict(zip(players, np.arange(len(players)) + 1))
        plots = dict()
        annots = dict()

        for p in players:
            (plots[p],) = ax.plot([], [], c=color, alpha=alpha, ls="--", zorder=0)

            player_num = player_dict[p] if anonymize else int(p[1:])
            annots[p] = ax.annotate(
                player_num,
                xy=traces.loc[0, [f"{p}_x", f"{p}_y"]],
                ha="center",
                va="center",
                color="w",
                fontsize=18,
                fontweight="bold",
                annotation_clip=False,
                zorder=3,
            )
            annots[p].set_animated(True)

        return traces, sizes, scat, plots, annots

    @staticmethod
    def animate_players(
        t: int,
        inplay_records: pd.DataFrame,
        traces: pd.DataFrame,
        sizes: np.ndarray,
        scat: collections.PatchCollection,
        plots: Dict[str, lines.Line2D],
        annots: Dict[str, text.Annotation],
    ):
        x = traces[traces.columns[0::2]].values
        y = traces[traces.columns[1::2]].values
        scat.set_offsets(np.stack([x[t], y[t]]).T)

        if type(sizes) == np.ndarray:
            scat.set_sizes(sizes[t])

        for p in plots.keys():
            inplay_start = inplay_records.at[p, "start_idx"]
            inplay_end = inplay_records.at[p, "end_idx"]
            if t >= inplay_start:
                if t <= inplay_end:
                    t_from = max(t - 9, inplay_start)
                    plots[p].set_data(traces.loc[t_from:t, f"{p}_x"], traces.loc[t_from:t, f"{p}_y"])
                    annots[p].set_position(traces.loc[t, [f"{p}_x", f"{p}_y"]])
                elif t == inplay_end + 1:
                    plots[p].set_alpha(0)
                    annots[p].set_alpha(0)

    @staticmethod
    def plot_ball(xy: pd.DataFrame, ax=axes.Axes, color="w", edgecolor="k", marker="o", show_path=True):
        x = xy.values[:, 0]
        y = xy.values[:, 1]
        scat = ax.scatter(x[0], y[0], s=300, c=color, edgecolors=edgecolor, marker=marker, zorder=4)

        if show_path:
            pathcolor = "k" if color == "w" else color
            (plot,) = ax.plot([], [], pathcolor, zorder=3)
        else:
            plot = None

        return x, y, scat, plot

    @staticmethod
    def animate_ball(
        t: int,
        x: np.ndarray,
        y: np.ndarray,
        scat: collections.PatchCollection,
        plot: lines.Line2D = None,
    ):
        scat.set_offsets(np.array([x[t], y[t]]))

        if plot is not None:
            t_from = max(t - 49, 0)
            plot.set_data(x[t_from : t + 1], y[t_from : t + 1])

    @staticmethod
    def plot_focus(xy: pd.DataFrame, ax=axes.Axes, color="purple"):
        x = xy.values[:, 0]
        y = xy.values[:, 1]
        center_scat = ax.scatter(x[0], y[0], s=300, c=color, marker="+", linewidths=2, zorder=4)
        edge_scat = ax.scatter(x[0], y[0], s=600, c="None", edgecolors=color, marker="s", linewidths=2, zorder=4)
        return x, y, center_scat, edge_scat

    @staticmethod
    def animate_focus(
        t: int,
        x: np.ndarray,
        y: np.ndarray,
        center_scat: collections.PatchCollection,
        edge_scat: collections.PatchCollection,
    ):
        center_scat.set_offsets(np.array([x[t], y[t]]))
        edge_scat.set_offsets(np.array([x[t], y[t]]))

    @staticmethod
    def plot_intent(ax=axes.Axes, patch_size=3):
        patch = Rectangle(xy=(0, 0), width=patch_size, height=patch_size, color="lime", alpha=0, zorder=0)
        ax.add_patch(patch)
        return patch

    @staticmethod
    def animate_intent(t: int, intents: np.array, patch=Rectangle, patch_size=3):
        patch.set_xy([intents[t, 0] - patch_size / 2, intents[t, 1] - patch_size / 2])
        patch.set_alpha(0.5)

    def plot_init(self, ax: axes.Axes, trace_key: str):
        traces = self.trace_dict[trace_key].iloc[(self.play_speed - 1) :: self.play_speed].copy()
        traces = traces.dropna(axis=1, how="all").reset_index(drop=True)
        xy_cols = [c for c in traces.columns if c.endswith("_x") or c.endswith("_y")]

        if self.rotate_pitch:
            traces[xy_cols[0::2]] = 108 - traces[xy_cols[0::2]]
            traces[xy_cols[1::2]] = 72 - traces[xy_cols[1::2]]

        inplay_records = []
        for c in xy_cols[::2]:
            inplay_idx = traces[traces[c].notna()].index
            inplay_records.append([c[:-2], inplay_idx[0], inplay_idx[-1]])
        inplay_records = pd.DataFrame(inplay_records, columns=["object", "start_idx", "end_idx"]).set_index("object")

        team1_traces = traces[[c for c in xy_cols if c.startswith("A")]].fillna(-100)
        team2_traces = traces[[c for c in xy_cols if c.startswith("B")]].fillna(-100)

        if trace_key == "main" and self.poss_probs is not None:
            probs = self.poss_probs
            if probs.shape[1] == 2:  # team_poss
                probs = probs.fillna(0.5).values[(self.play_speed - 1) :: self.play_speed]
                team1_sizes = np.repeat(probs[:, [0]] * 500 + 500, team1_traces.shape[1], axis=1)
                team2_sizes = np.repeat(probs[:, [1]] * 500 + 500, team2_traces.shape[1], axis=1)
            else:  # player_poss
                n_players = team1_traces.shape[1] // 2
                probs = probs.dropna(axis=1, how="all")
                probs = probs.fillna(1 / probs.shape[1]).values[(self.play_speed - 1) :: self.play_speed]
                team1_sizes = probs[:, :n_players] * 1500 + 500
                team2_sizes = probs[:, n_players : n_players * 2] * 1500 + 500

        else:
            team1_sizes = 750
            team2_sizes = 750

        alpha = 1 if trace_key == "main" else 0.5
        team1_args = self.plot_players(team1_traces, ax, team1_sizes, alpha, self.anonymize)
        team2_args = self.plot_players(team2_traces, ax, team2_sizes, alpha, self.anonymize)

        ball_args = None
        if "ball_x" in traces.columns and traces["ball_x"].notna().any():
            ball_xy = traces[["ball_x", "ball_y"]]
            if trace_key == "main":
                ball_args = TraceAnimator.plot_ball(ball_xy, ax, "w", "k", "o")
            else:
                ball_args = TraceAnimator.plot_ball(ball_xy, ax, trace_key, None, "*")

        focus_args = None
        if self.show_focus and "focus_x" in traces.columns and traces["focus_x"].notna().any():
            focus_xy = traces[["focus_x", "focus_y"]]
            focus_args = TraceAnimator.plot_focus(focus_xy, ax)

        # if "intent_x" in traces.columns and traces["intent_x"].notna().any():
        #     intents = traces[["intent_x", "intent_y"]].values
        #     patch = TraceHelper.plot_intent(ax)
        #     intent_args = [intents, patch]

        self.trace_dict[trace_key] = traces
        self.anim_args.loc[trace_key] = [inplay_records, team1_args, team2_args, ball_args, focus_args]

    def run(self):
        fig, ax = plt.subplots(figsize=(20.8, 14.4))
        mps.field("white", fig, ax, show=False)

        for key in self.trace_dict.keys():
            self.plot_init(ax, key)

        traces = self.trace_dict["main"]

        if self.show_times:
            time_texts = traces["time"].apply(lambda x: f"{int(x // 60):02d}:{x % 60:04.1f}").values
            time_annot = ax.text(0, 73, time_texts[0], fontsize=20, ha="left", va="bottom")
            time_annot.set_animated(True)

        if self.show_episodes:
            episode_texts = traces["episode"].apply(lambda x: f"Episode {x}")
            episode_texts = np.where(episode_texts == "Episode 0", "", episode_texts)
            episode_annot = ax.text(93, 73, episode_texts[0], fontsize=20, ha="left", va="bottom")
            episode_annot.set_animated(True)

        if self.show_events:
            if "event_type" in traces.columns:  # (for Metrica data)
                event_texts = traces.apply(lambda x: f"{x['event_type']} by {x['event_player']}", axis=1)
                event_texts = np.where(event_texts == "nan by nan", "", event_texts)
            else:  # if "event_types" in traces.columns: (for GPS-event data)
                event_texts = traces["event_types"].fillna(method="ffill")
                event_texts = np.where(event_texts.isna(), "", event_texts)

            event_annot = ax.text(15, 73, str(event_texts[0]), fontsize=20, ha="left", va="bottom")
            event_annot.set_animated(True)

        if self.annot_cols is not None:
            text_dict = {}
            annot_dict = {}
            for i, col in enumerate(self.annot_cols):
                text_dict[col] = f"{col}: " + np.where(traces[col].isna(), "", traces[col].astype(str))
                annot_dict[col] = ax.text(i * 54, -1, str(text_dict[col][0]), fontsize=20, ha="left", va="top")
                annot_dict[col].set_animated(True)

        def animate(t):
            for key in self.trace_dict.keys():
                inplay_records = self.anim_args.at[key, "inplay_records"]
                team1_args = self.anim_args.at[key, "team1"]
                team2_args = self.anim_args.at[key, "team2"]
                ball_args = self.anim_args.at[key, "ball"]
                focus_args = self.anim_args.at[key, "focus"]

                if team1_args is not None:
                    TraceAnimator.animate_players(t, inplay_records, *team1_args)
                if team2_args is not None:
                    TraceAnimator.animate_players(t, inplay_records, *team2_args)
                if ball_args is not None:
                    TraceAnimator.animate_ball(t, *ball_args)
                if focus_args is not None:
                    TraceAnimator.animate_focus(t, *focus_args)

            if self.show_times:
                time_annot.set_text(str(time_texts[t]))

            if self.show_episodes:
                episode_annot.set_text(str(episode_texts[t]))

            if self.show_events:
                event_annot.set_text(event_texts[t])

            if self.annot_cols is not None:
                for col in self.annot_cols:
                    annot_dict[col].set_text(str(text_dict[col][t]))

        frames = min(self.max_frames, traces.shape[0])
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000 / self.fps)
        plt.close(fig)

        return anim
