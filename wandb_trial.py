#!/usr/bin/env python3
"""
One W&B sweep trial: train a LoRA adapter with the sweep's values, evaluate it, report back.

`wandb agent` starts this script once per trial (see sweep.yaml); the values to try and
the fixed settings both come from wandb.config. Training curves go to the W&B run
through TRL: reward / reward_std, kl, entropy, and the tier and novelty metrics from
reward.py, plus the `completions` table every step and the `samples` table every
`sample_every` steps. The trained model is evaluated in memory with the same loop and
settings as evaluate.py; per-step eval curves go under eval/, and `final_unique_commands`
(the sweep's metric) is logged at the end with the other summaries and the discovery
curve next to the base model's.

Files per trial: <out_dir>/<run id>/ holds the adapter, samples.jsonl and eval/.
"""

import csv
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import wandb

from evaluate import evaluate_loaded, plot
from reward import TIERS
from train_grpo import build_parser, train


def main() -> int:
    run = wandb.init()
    # Copy the sweep's values now: TRL's W&B callback later writes its own settings into run.config,
    # and some names overlap (TrainingArguments has eval_steps, seed, ...).
    c = SimpleNamespace(**{"num_generations": 8, "sample_every": 10, "base_eval": None, **dict(run.config)})
    out = Path(c.out_dir) / run.id

    args = build_parser().parse_args([
        "--model", c.model, "--histories", c.histories, "--output-dir", str(out),
        "--target", c.target, "--interface", c.interface,
        "--lora", "--max-steps", str(c.trial_steps), "--save-steps", str(10 ** 9), "--seed", str(c.seed),
        "--lr", str(c.lr), "--beta", str(c.beta), "--temperature", str(c.temperature),
        "--num-generations", str(c.num_generations),
        "--report-to", "wandb", "--log-completions", "--sample-every", str(c.sample_every),
    ])
    trainer = train(args)

    model = trainer.model.merge_and_unload().eval()
    tok = trainer.processing_class
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    eval_dir = out / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_args = SimpleNamespace(target=c.target, interface=c.interface, episodes=c.eval_episodes,
                                steps=c.eval_length, temperature=1.0, max_new_tokens=64,
                                batch_size=c.batch_size, seed=c.seed, out_dir=eval_dir)
    rows = evaluate_loaded(model, tok, f"{c.model} + {run.id}", "trained", eval_args)

    with open(eval_dir / "eval_steps.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    for r in rows:
        wandb.log({"eval/step": r["step"], "eval/unique_commands": r["mean_unique_commands"],
                   "eval/plausible_unique_commands": r["mean_plausible_unique_commands"],
                   "eval/reward_mean": r["mean_reward"], "eval/reward_std": r["std_reward"],
                   **{f"eval/{t}_rate": r[f"{t}_rate"] for t in TIERS}})

    labels, plot_rows, summary = ["trained"], list(rows), {}
    base_csv = Path(c.base_eval) if c.base_eval else None
    if base_csv and base_csv.exists():
        with open(base_csv, encoding="utf-8") as f:
            base = [{k: (v if k in ("label", "model") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
        labels, plot_rows = ["base", "trained"], base + plot_rows
        summary["base_final_unique_commands"] = base[-1]["mean_unique_commands"]
        summary["gain_over_base"] = rows[-1]["mean_unique_commands"] - base[-1]["mean_unique_commands"]
    plot(plot_rows, labels, eval_dir / "discovery_curve.png")

    with open(eval_dir / "trained_histories.jsonl", encoding="utf-8") as f:
        first = json.loads(f.readline())["history"]
    examples = wandb.Table(columns=["command", "parameters"],
                           data=[[h["command"], json.dumps(h["parameters"])] for h in first])

    wandb.log({
        **summary,
        "final_unique_commands": rows[-1]["mean_unique_commands"],
        "final_plausible_unique_commands": rows[-1]["mean_plausible_unique_commands"],
        "eval_reward_mean": statistics.mean(r["mean_reward"] for r in rows),
        **{f"{t}_rate": statistics.mean(r[f"{t}_rate"] for r in rows) for t in TIERS},
        "eval/discovery_curve": wandb.Image(str(eval_dir / "discovery_curve.png")),
        "eval/example_list": examples,
    })
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
