#!/usr/bin/env python3
"""
Random hyperparameter sweep for train_grpo.py, run until a time budget is used up.

1. Builds one shared histories file with the base model (skipped if it already exists).
2. Evaluates the base model once, as the reference.
3. Repeats until --hours is up: sample a config from RANGES, train a LoRA adapter for
   --trial-steps updates, evaluate it, append a row to <out-dir>/trials.csv.
   A new trial only starts if the previous one would still fit in the remaining time.

Every trial uses the same histories, the same evaluation settings (temperature 1.0,
same seed), and only differs in the sampled values, so the rows are comparable.
Rerunning the same command continues after the last finished trial.

Each trial's merged model is deleted after its evaluation to save disk (the LoRA adapter
is kept in <out-dir>/trial_NNN/); pass --keep-models to keep them.

    python sweep.py --model Qwen/Qwen2.5-3B-Instruct --hours 20 --out-dir sweeps/qwen3b
"""

import argparse
import csv
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Sampled per trial. Learning rate on a log scale.
RANGES = {
    "lr": (1e-6, 1e-4, "log"),
    "beta": (0.0, 0.1, "linear"),
    "temperature": (0.8, 1.3, "linear"),
}

FIELDS = ["trial", "lr", "beta", "temperature", "status", "minutes",
          "final_unique_commands", "broken_rate", "new_structure_rate"]


def sample(rng):
    cfg = {}
    for name, (lo, hi, scale) in RANGES.items():
        if scale == "log":
            cfg[name] = float(f"{math.exp(rng.uniform(math.log(lo), math.log(hi))):.3g}")
        else:
            cfg[name] = round(rng.uniform(lo, hi), 3)
    return cfg


def run(cmd, log_path):
    """Run a script with its output going to log_path. Returns True on success."""
    print("  $", " ".join(map(str, cmd)), flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        return subprocess.run([sys.executable, *map(str, cmd)], stdout=log, stderr=subprocess.STDOUT).returncode == 0


def eval_summary(eval_dir, label):
    """Final mean unique commands, and broken / new-command rates averaged over all steps."""
    with open(eval_dir / "eval_steps.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] == label]
    mean = lambda k: sum(float(r[k]) for r in rows) / len(rows)
    return {"final_unique_commands": round(float(rows[-1]["mean_unique_commands"]), 2),
            "broken_rate": round(mean("broken_rate"), 3),
            "new_structure_rate": round(mean("new_structure_rate"), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base model, HF id")
    ap.add_argument("--hours", type=float, required=True, help="time budget for the whole sweep")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--history-steps", type=int, default=100, help="list length when building training prompts")
    ap.add_argument("--history-episodes", type=int, default=64)
    ap.add_argument("--trial-steps", type=int, default=150, help="training updates per trial")
    ap.add_argument("--eval-episodes", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16, help="episodes generated at once (collection and eval)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-models", action="store_true", help="keep each trial's merged model")
    args = ap.parse_args()

    deadline = time.time() + args.hours * 3600
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    eval_args = ["--episodes", args.eval_episodes, "--steps", args.eval_steps,
                 "--batch-size", args.batch_size, "--seed", args.seed]

    histories = out / "histories.jsonl"
    if not histories.exists():
        print("Building shared histories with the base model", flush=True)
        tmp = out / "histories.partial.jsonl"
        if not run([here / "collect_histories.py", "--model", args.model, "--steps", args.history_steps,
                    "--episodes", args.history_episodes, "--batch-size", args.batch_size, "--out", tmp],
                   out / "histories.log"):
            print(f"collect_histories.py failed, see {out / 'histories.log'}")
            return 1
        tmp.rename(histories)

    base_eval = out / "base_eval"
    if not (base_eval / "eval_steps.csv").exists():
        print("Evaluating the base model", flush=True)
        if not run([here / "evaluate.py", "--models", args.model, "--labels", "base", "--out-dir", base_eval,
                    *eval_args], out / "base_eval.log"):
            print(f"evaluate.py failed, see {out / 'base_eval.log'}")
            return 1
    print(f"Base model: {eval_summary(base_eval, 'base')}", flush=True)

    trials_csv = out / "trials.csv"
    done = 0
    if trials_csv.exists():
        with open(trials_csv, encoding="utf-8") as f:
            done = sum(1 for _ in csv.DictReader(f))
    else:
        with open(trials_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    rng = random.Random(args.seed)
    for _ in range(done):  # skip configs already tried, so a rerun continues the same sequence
        sample(rng)

    last_minutes = None
    trial = done
    while True:
        remaining = (deadline - time.time()) / 60
        if last_minutes is not None and remaining < last_minutes:
            print(f"Stopping: {remaining:.0f} min left, last trial took {last_minutes:.0f} min")
            break
        if remaining <= 0:
            break

        cfg = sample(rng)
        name = f"trial_{trial:03d}"
        tdir = out / name
        print(f"\n[{name}] {cfg}  ({remaining:.0f} min left)", flush=True)
        start = time.time()

        ok = run([here / "train_grpo.py", "--model", args.model, "--histories", histories, "--output-dir", tdir,
                  "--lora", "--max-steps", args.trial_steps, "--save-steps", 10 ** 9, "--seed", args.seed,
                  "--lr", cfg["lr"], "--beta", cfg["beta"], "--temperature", cfg["temperature"]],
                 out / f"{name}_train.log")
        row = {"trial": name, **cfg, "status": "train_failed"}
        if ok:
            ok = run([here / "evaluate.py", "--models", tdir / "merged", "--labels", name,
                      "--out-dir", tdir / "eval", *eval_args], out / f"{name}_eval.log")
            row["status"] = "ok" if ok else "eval_failed"
            if ok:
                row.update(eval_summary(tdir / "eval", name))
        if not args.keep_models:
            shutil.rmtree(tdir / "merged", ignore_errors=True)

        last_minutes = (time.time() - start) / 60
        row["minutes"] = round(last_minutes, 1)
        with open(trials_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"[{name}] {row['status']} in {last_minutes:.0f} min: "
              f"{row.get('final_unique_commands', '-')} unique commands", flush=True)
        trial += 1

    with open(trials_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    rows.sort(key=lambda r: float(r["final_unique_commands"]), reverse=True)
    print(f"\nBase model: {eval_summary(base_eval, 'base')}")
    print("Best trials:")
    for r in rows[:5]:
        print(f"  {r['trial']}: {r['final_unique_commands']} unique, {float(r['broken_rate']):.0%} broken  "
              f"(lr={r['lr']}, beta={r['beta']}, temperature={r['temperature']})")
    print(f"All results: {trials_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
