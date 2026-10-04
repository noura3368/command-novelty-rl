#!/usr/bin/env python3
"""
Compare how many different commands models find over a long run.

Each model explores exactly as in collect_histories.py (same prompt, same settings,
same seed): --episodes independent lists, one command per list per step, each new
command appended to its list. Per step it records, averaged over the lists:

    unique commands     distinct `command` values in the list (the main number)
    plausible unique    the same, counting only commands that pass reward.plausible_syntax
    entries             list length (distinct command + parameters pairs)
    tier rates          share of replies that were broken / repeat / new_value / new_structure,
                        judged against the list before the reply (same rules as reward.py)
    reward              mean and sd of the reward each reply would get as a group of one

Writes <out-dir>/eval_steps.csv, <out-dir>/<label>_histories.jsonl (final lists),
and <out-dir>/discovery_curve.png.

    python evaluate.py --models Qwen/Qwen2.5-7B-Instruct runs/round0/merged \\
        --labels base round0 --steps 100 --out-dir eval/round0
"""

import argparse
import csv
import gc
import json
import statistics
from pathlib import Path

import torch

from collect_histories import load_model, run_episodes
from reward import TIERS, plausible_syntax, score_group

# Categorical slots in fixed order (light surface); a model keeps its slot by position.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a63", "#e6e5e0"


def default_label(model_id: str) -> str:
    p = Path(model_id)
    return p.parent.name if p.name == "merged" else p.name


def evaluate_model(model_id, label, args):
    model, tok = load_model(model_id)
    rows = evaluate_loaded(model, tok, model_id, label, args)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def evaluate_loaded(model, tok, model_id, label, args):
    """Run the loop with an already loaded model; write <label>_histories.jsonl and return the per-step rows."""
    torch.manual_seed(args.seed)
    rows, final = [], None
    for step, before, replies, after in run_episodes(model, tok, args.target, args.interface, args.episodes,
                                                     args.steps, args.temperature, args.max_new_tokens,
                                                     args.batch_size):
        scored = [score_group([r], b)[0] for r, b in zip(replies, before)]
        tiers = [t for t, _, _ in scored]
        rewards = [r for _, r, _ in scored]
        unique = [len({x["command"] for x in h}) for h in after]
        row = {
            "label": label, "model": model_id, "step": step + 1,
            "mean_unique_commands": statistics.mean(unique),
            "std_unique_commands": statistics.pstdev(unique),
            "mean_plausible_unique_commands": statistics.mean(
                len({x["command"] for x in h if plausible_syntax(x["command"])}) for h in after),
            "mean_entries": statistics.mean(len(h) for h in after),
            "mean_reward": statistics.mean(rewards),
            "std_reward": statistics.pstdev(rewards),
        }
        for t in TIERS:
            row[f"{t}_rate"] = tiers.count(t) / len(tiers)
        rows.append(row)
        final = after
        print(f"[{label}] step {step + 1}/{args.steps}: {row['mean_unique_commands']:.2f} unique commands, "
              f"{row['broken_rate']:.0%} broken", flush=True)

    with open(args.out_dir / f"{label}_histories.jsonl", "w", encoding="utf-8") as f:
        for e, h in enumerate(final):
            f.write(json.dumps({"episode": e, "history": h}) + "\n")
    return rows


def plot(rows, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, label in enumerate(labels):
        r = [x for x in rows if x["label"] == label]
        steps = [x["step"] for x in r]
        mean = [x["mean_unique_commands"] for x in r]
        std = [x["std_unique_commands"] for x in r]
        color = COLORS[i % len(COLORS)]
        ax.fill_between(steps, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(steps, mean, color=color, linewidth=2, label=label)
        if len(labels) <= 4:
            ax.annotate(f"{label}  {mean[-1]:.1f}", (steps[-1], mean[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=TEXT)

    ax.set_title("Unique commands found per list (mean ± 1 sd across lists)", loc="left", fontsize=11, color=TEXT)
    ax.set_xlabel("Step", color=MUTED)
    ax.set_ylabel("Unique commands", color=MUTED)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if len(labels) > 1:
        ax.legend(frameon=False, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="HF model ids or local checkpoints, e.g. base then trained")
    ap.add_argument("--labels", nargs="+", help="names for the models in the CSV and plot (default: from the path)")
    ap.add_argument("--target", default="KORAD KA3005P Power Supply")
    ap.add_argument("--interface", default="RS232 interface")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16, help="episodes generated at once on the GPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    labels = args.labels or [default_label(m) for m in args.models]
    if len(labels) != len(args.models) or len(set(labels)) != len(labels):
        ap.error("--labels must give one distinct name per model")
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_id, label in zip(args.models, labels):
        rows += evaluate_model(model_id, label, args)

    with open(args.out_dir / "eval_steps.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, labels, args.out_dir / "discovery_curve.png")

    print("\nAfter the last step:")
    for label in labels:
        r = [x for x in rows if x["label"] == label]
        broken = statistics.mean(x["broken_rate"] for x in r)
        print(f"  {label}: {r[-1]['mean_unique_commands']:.2f} unique commands per list, {broken:.0%} broken overall")
    print(f"Results in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
