#!/usr/bin/env python3
"""
Figures on reward hacking in saved eval lists (*_histories.jsonl), judged by the reward.py rules.

Each list's distinct names are replayed in order and given the tier score_group would give them
against the names before them: wordy (2+ connector words), repeat (same command words as an
earlier name) or new. Within each sweep (3b-v2, 7b-v2, ...) the base model's eval is kept as a
reference. Hyperparameters come from the trials' W&B config.yaml, as in hack_report.py.

    python plot_hacks.py sweeps/*-v2/base_eval/*_histories.jsonl sweeps/*-v2/wandb/*/eval/trained_histories.jsonl \\
        --out-dir sweeps/figures-v2

Writes to --out-dir:
  composition.png   per trial, the mean distinct names per list (the v2 score) split into new / wordy / repeat
  score_vs_new.png  that score against the names that stay new under the v4 reward
  hyperparams.png   share of names that are wordy or repeats, against lr, beta and temperature
  over_list.png     that share by position in the list, for the TOP_TRIALS most hacking trials of each sweep
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from hack_report import classify, hyperparams, sweep_and_run

TIERS = ("new", "wordy", "repeat")
BIN = 15            # list positions per point in over_list.png
TOP_TRIALS = 3
LABEL_GAP = 5      # percentage points between end labels in over_list.png

# Reference palette (dataviz skill), light mode; the first three slots pass every colour check together.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a63", "#e6e5e0"


def load(path: Path) -> Dict:
    sweep, run = sweep_and_run(path)
    with open(path, encoding="utf-8") as f:
        lists = [list(dict.fromkeys(x["command"] for x in json.loads(line)["history"])) for line in f]
    tiers = [classify(ns) for ns in lists]
    names = sum(map(len, lists))
    hacked = sum(t != "new" for ts in tiers for t in ts)
    bins = defaultdict(list)
    for ts in tiers:
        for i, t in enumerate(ts):
            bins[i // BIN].append(t != "new")
    return {
        "sweep": sweep, "trial": run, "hp": hyperparams(path),
        "names": statistics.mean(map(len, lists)),
        **{t: statistics.mean(ts.count(t) for ts in tiers) for t in TIERS},
        "hack_share": hacked / names if names else 0.0,
        "over_list": [(b * BIN + BIN / 2, 100 * sum(v) / len(v)) for b, v in sorted(bins.items())],
    }


def style(ax, plt):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def size_colors(sweeps):
    return {s: COLORS[i % len(COLORS)] for i, s in enumerate(sweeps)}


def plot_composition(trials, sweeps, path, plt):
    fig, axes = plt.subplots(1, len(sweeps), figsize=(5.5 * len(sweeps), 7.5), facecolor=SURFACE, sharex=True)
    axes = [axes] if len(sweeps) == 1 else axes
    xmax = max(t["names"] for t in trials) * 1.25
    for ax, sweep in zip(axes, sweeps):
        style(ax, plt)
        rows = sorted((t for t in trials if t["sweep"] == sweep), key=lambda t: (t["trial"] == "base", -t["new"]))
        rows.reverse()                                  # barh draws bottom-up: best trial on top, base at the bottom
        y = range(len(rows))
        left = [0.0] * len(rows)
        for color, tier in zip(COLORS, TIERS):
            vals = [t[tier] for t in rows]
            ax.barh(y, vals, left=left, height=0.72, color=color, edgecolor=SURFACE, linewidth=1, label=tier)
            left = [a + b for a, b in zip(left, vals)]
        for yi, t in zip(y, rows):
            ax.annotate(f"{100 * t['hack_share']:.0f}% hacked", (t["names"], yi), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=MUTED)
        ax.set_yticks(list(y), [t["trial"] for t in rows], fontsize=8, color=TEXT, family="monospace")
        ax.set_xlim(0, xmax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(sweep, loc="left", fontsize=10, color=TEXT)
        ax.set_xlabel("Distinct names per list (mean of the eval lists)", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, labelcolor=TEXT, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.04))
    fig.suptitle("What each trial's score is made of, judged by the v4 reward (hacked = wordy + repeat)", x=0.01, ha="left",
                 fontsize=11, color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def plot_score_vs_new(trials, sweeps, path, plt):
    colors = size_colors(sweeps)
    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor=SURFACE)
    style(ax, plt)
    top = max(t["names"] for t in trials) * 1.05
    ax.plot([0, top], [0, top], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("no hacking", (top * 0.5, top * 0.5), xytext=(-4, 6), textcoords="offset points",
                fontsize=8, color=MUTED, rotation=45, ha="right")
    labels = []
    for sweep in sweeps:
        rows = [t for t in trials if t["sweep"] == sweep]
        for base in (False, True):
            r = [t for t in rows if (t["trial"] == "base") == base]
            ax.scatter([t["names"] for t in r], [t["new"] for t in r], s=70 if base else 55,
                       marker="D" if base else "o", color=colors[sweep], edgecolor=SURFACE, linewidth=1.5,
                       label=f"{sweep} base" if base else sweep, zorder=3)
        trained = [t for t in rows if t["trial"] != "base"]
        notable = sorted(trained, key=lambda t: t["new"] - t["names"])[:2] + [max(trained, key=lambda t: t["new"])]
        # [label y, point x, point y, name]; label y starts at the point and may be pushed up below
        labels += [[t["new"], t["names"], t["new"], t["trial"]] for t in {id(t): t for t in notable}.values()]
    labels.sort()
    for lower, upper in zip(labels, labels[1:]):      # keep labels at least 3% of the axis apart vertically
        upper[0] = max(upper[0], lower[0] + top * 0.03)
    for y, x, y0, name in labels:
        ax.annotate(name, (x, y0), xytext=(x + top * 0.02, y), textcoords="data", va="center",
                    fontsize=7.5, color=TEXT, family="monospace",
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6) if abs(y - y0) > 0.5 else None)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Distinct names per list (the v2 score)", color=MUTED, fontsize=9)
    ax.set_ylabel("New commands per list under the v4 reward", color=MUTED, fontsize=9)
    ax.set_title("Score with and without hacks (distance below the line = wordy or repeat names)", loc="left",
                 fontsize=10, color=TEXT)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def plot_hyperparams(trials, sweeps, path, plt):
    import matplotlib
    import matplotlib.ticker
    colors = size_colors(sweeps)
    params = [("lr", "Learning rate", True), ("beta", "beta (KL penalty)", False), ("temperature", "Training temperature", False)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor=SURFACE, sharey=True)
    for ax, (key, label, log) in zip(axes, params):
        style(ax, plt)
        for sweep in sweeps:
            r = [t for t in trials if t["sweep"] == sweep and t["hp"].get(key)]
            ax.scatter([float(t["hp"][key]) for t in r], [100 * t["hack_share"] for t in r], s=55,
                       color=colors[sweep], edgecolor=SURFACE, linewidth=1.5, label=sweep, zorder=3)
        if log:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1e-5, 2e-5, 5e-5, 1e-4]))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0e}".replace("e-0", "e-")))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(label, color=MUTED, fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Names that are wordy or repeats (%)", color=MUTED, fontsize=9)
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="upper left")
    fig.suptitle("Hacking against the searched hyperparameters (one dot per trial)", x=0.01, ha="left",
                 fontsize=11, color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def plot_over_list(trials, sweeps, path, plt):
    fig, axes = plt.subplots(1, len(sweeps), figsize=(5.5 * len(sweeps), 4), facecolor=SURFACE, sharey=True)
    axes = [axes] if len(sweeps) == 1 else axes
    for ax, sweep in zip(axes, sweeps):
        style(ax, plt)
        trained = [t for t in trials if t["sweep"] == sweep and t["trial"] != "base"]
        ends = []
        for color, t in zip(COLORS, sorted(trained, key=lambda t: -t["hack_share"])[:TOP_TRIALS]):
            xs, ys = zip(*t["over_list"])
            ax.plot(xs, ys, color=color, linewidth=2, label=t["trial"])
            ends.append([ys[-1], xs[-1], t["trial"]])
        ends.sort()
        for lower, upper in zip(ends, ends[1:]):      # keep end labels at least LABEL_GAP points of y apart
            upper[0] = max(upper[0], lower[0] + LABEL_GAP)
        for y, x, name in ends:
            ax.annotate(name, (x, y), xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=TEXT, family="monospace", annotation_clip=False)
        ax.set_title(f"{sweep}: the {TOP_TRIALS} most hacking trials", loc="left", fontsize=10, color=TEXT)
        ax.set_xlabel("Position in the list (names in the order they were added)", color=MUTED, fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 100)
        ax.margins(x=0.12)
        ax.legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="lower right")
    axes[0].set_ylabel(f"Wordy or repeat (%, per {BIN} positions)", color=MUTED, fontsize=9)
    fig.suptitle("When hacking starts within a list", x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="*_histories.jsonl from evaluate.py or a sweep trial's eval/")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = [load(f) for f in args.files]
    sweeps = sorted({t["sweep"] for t in trials})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_composition(trials, sweeps, args.out_dir / "composition.png", plt)
    plot_score_vs_new(trials, sweeps, args.out_dir / "score_vs_new.png", plt)
    plot_hyperparams(trials, sweeps, args.out_dir / "hyperparams.png", plt)
    plot_over_list(trials, sweeps, args.out_dir / "over_list.png", plt)
    print(f"wrote 4 figures for {len(trials)} evals to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
