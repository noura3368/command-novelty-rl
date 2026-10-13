#!/usr/bin/env python3
"""
Compare how many different commands models find over a long run.

Each model explores exactly as in collect_histories.py (same prompt, same settings,
same seed): --episodes independent lists, one command per list per step. A reply is appended
to its list if it parses and its name is new under --list-rule (parameters are never compared):

    exact           (default, v5) no entry has exactly the same command name; nothing else is
                    checked, so every different name the model produces enters the list
    command_words   v4's rule: not wordy and no entry with the same reward.command_words

Per step it records, averaged over the lists:

    new commands        entries in the list, so the list length: what the v5 reward pays for. The main number
    clean commands      the same lists recounted with clean_histories.py's rules (no wordy names, no
                        repeats once case, separators and numbers are ignored): what is left without hacks
    hack share          1 - clean / new over all entries of all lists: the share of the lists that is hacked
    unique commands     distinct `command` strings in the list (equal to new commands under `exact`)
    plausible unique    the same, counting only commands that pass reward.plausible_syntax
    real commands       distinct names in the documented KA3005P set (reward.ka3005p_command)
    name length         median length of the names in the lists, in characters
    tier rates          share of replies that were broken / repeat / new_structure,
                        judged against the list before the reply by reward.py
    reward              mean and sd of the reward each reply gets

Writes <out-dir>/eval_steps.csv, <out-dir>/<label>_histories.jsonl (final lists),
and <out-dir>/discovery_curve.png (new commands solid, clean commands dashed).

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

from clean_histories import clean
from collect_histories import LIST_RULES, load_model, run_episodes
from reward import TIERS, ka3005p_command, plausible_syntax, score_group

# Categorical slots in fixed order (light surface); a model keeps its slot by position.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a63", "#e6e5e0"


MAIN = "mean_new_commands"     # the eval's headline column: list length
CLEAN = "mean_clean_commands"  # the same lists without hacks (clean_histories.clean)


def hack_share(new: list, clean_counts: list) -> float:
    """Share of all list entries that clean_histories.clean removes; 0 for empty lists."""
    return 1 - sum(clean_counts) / sum(new) if sum(new) else 0.0


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
    list_rule = getattr(args, "list_rule", "exact")
    rows, final = [], None
    for step, before, replies, after in run_episodes(model, tok, args.target, args.interface, args.episodes,
                                                     args.steps, args.temperature, args.max_new_tokens,
                                                     args.batch_size, list_rule):
        scored = [score_group([r], b)[0] for r, b in zip(replies, before)]
        tiers = [t for t, _, _ in scored]
        rewards = [r for _, r, _ in scored]
        unique = [len({x["command"] for x in h}) for h in after]
        new = [len(h) for h in after]
        kept = [len(clean(h)[0]) for h in after]
        row = {"label": label, "model": model_id, "list_rule": list_rule, "seed": args.seed, "step": step + 1,
               MAIN: statistics.mean(new), "std_new_commands": statistics.pstdev(new),
               CLEAN: statistics.mean(kept), "std_clean_commands": statistics.pstdev(kept),
               "hack_share": hack_share(new, kept)}
        row.update({
            "mean_unique_commands": statistics.mean(unique),
            "std_unique_commands": statistics.pstdev(unique),
            "mean_plausible_unique_commands": statistics.mean(
                len({x["command"] for x in h if plausible_syntax(x["command"])}) for h in after),
            "mean_real_commands": statistics.mean(
                len({x["command"] for x in h if ka3005p_command(x["command"])}) for h in after),
            "median_name_length": statistics.median([len(x["command"]) for h in after for x in h] or [0]),
            "mean_reward": statistics.mean(rewards),
            "std_reward": statistics.pstdev(rewards),
        })
        for t in TIERS:
            row[f"{t}_rate"] = tiers.count(t) / len(tiers)
        rows.append(row)
        final = after
        mem = f", {torch.cuda.max_memory_reserved() / 2**30:.1f} GB peak reserved" if torch.cuda.is_available() else ""
        print(f"[{label}] step {step + 1}/{args.steps}: {row[MAIN]:.2f} new commands, {row[CLEAN]:.2f} clean "
              f"({row['hack_share']:.0%} hacked), {row['broken_rate']:.0%} broken{mem}", flush=True)

    with open(args.out_dir / f"{label}_histories.jsonl", "w", encoding="utf-8") as f:
        for e, h in enumerate(final):
            f.write(json.dumps({"episode": e, "history": h}) + "\n")
    return rows


def load_eval(csv_path: Path):
    """(per-step rows, final summary) of a saved eval, e.g. a baseline for wandb_trial.py.

    The summary (new, clean, hack share) is counted from the final lists in the *_histories.jsonl
    next to the CSV when there is one, so it also works for CSVs written before v5. Those had no
    new/clean split: their mean_clean_commands was the list length, and is read as new commands.
    """
    with open(csv_path, encoding="utf-8") as f:
        rows = [{k: (v if k in ("label", "model", "list_rule") else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]
    if rows and MAIN not in rows[0]:
        for r in rows:
            r[MAIN], r["std_new_commands"] = r.pop(CLEAN), r.pop("std_clean_commands")
    summary = {"new": rows[-1][MAIN], "clean": rows[-1].get(CLEAN), "hack_share": rows[-1].get("hack_share")}
    lists_file = next(iter(sorted(csv_path.parent.glob("*_histories.jsonl"))), None)
    if lists_file is not None:
        with open(lists_file, encoding="utf-8") as f:
            lists = [json.loads(line)["history"] for line in f]
        new, kept = [len(h) for h in lists], [len(clean(h)[0]) for h in lists]
        summary = {"new": statistics.mean(new), "clean": statistics.mean(kept), "hack_share": hack_share(new, kept)}
    return rows, summary


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
        mean = [x[MAIN] for x in r]
        std = [x[MAIN.replace("mean_", "std_", 1)] for x in r]
        color = COLORS[i % len(COLORS)]
        ax.fill_between(steps, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(steps, mean, color=color, linewidth=2, label=label)
        if all(CLEAN in x for x in r):
            ax.plot(steps, [x[CLEAN] for x in r], color=color, linewidth=1.5, linestyle="--")
        if len(labels) <= 4:
            ax.annotate(f"{label}  {mean[-1]:.1f}", (steps[-1], mean[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=TEXT)

    ax.set_title(f"Commands per list, list rule {rows[0]['list_rule']} (mean ± 1 sd across lists; "
                 "dashed: without hacks)", loc="left", fontsize=11, color=TEXT)
    ax.set_xlabel("Step", color=MUTED)
    ax.set_ylabel("Commands in the list", color=MUTED)
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
    ap.add_argument("--list-rule", choices=LIST_RULES, default="exact",
                    help="which replies enter a list (see above); default exact: any new command name")
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
        print(f"  {label}: {r[-1][MAIN]:.2f} new commands per list, {r[-1][CLEAN]:.2f} without hacks "
              f"({r[-1]['hack_share']:.0%} hacked); {broken:.0%} broken overall")
    print(f"Results in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
