#!/usr/bin/env python3
"""
One W&B sweep trial: train a LoRA adapter with the sweep's values, evaluate it, report back.

`wandb agent` starts this script once per trial (see sweep.yaml); the values to try and
the fixed settings both come from wandb.config. Training curves go to the W&B run
through TRL: reward / reward_std, kl, entropy, and the tier and novelty metrics from
reward.py, plus the `completions` table every step and the `samples` table every
`sample_every` steps. The trained model is evaluated in memory with the same loop and
settings as evaluate.py; per-step eval curves go under eval/, and the final values
(`final_unique_commands`, `final_first3_commands`, ...; the sweep YAML names the one it
maximizes) are logged at the end with the other summaries and the discovery curve next
to the base model's. A `version` value in the sweep config (e.g. v3) is added as a run tag.

Files per trial: <out_dir>/<run id>/ holds the adapter, samples.jsonl and eval/.

A trial that finished training but failed during its eval (e.g. out of GPU memory) can be
evaluated afterwards from its saved adapter, into the same W&B run. The run gets the tag
`rescored` and summary `rescored: true`, so these scores can be told apart from the rest:

    python wandb_trial.py --rescore <entity>/<project>/<run id>
"""

import csv
import gc
import json
import statistics
import sys
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
    defaults = {"num_generations": 8, "sample_every": 10, "base_eval": None,
                "micro_batch": 4, "grad_accum": 8, "generation_batch": 16}
    c = SimpleNamespace(**{**defaults, **dict(run.config)})
    out = Path(c.out_dir) / run.id
    if getattr(c, "version", None):
        run.tags = tuple(sorted(set(run.tags) | {c.version}))

    args = build_parser().parse_args([
        "--model", c.model, "--histories", c.histories, "--output-dir", str(out),
        "--target", c.target, "--interface", c.interface,
        "--lora", "--max-steps", str(c.trial_steps), "--save-steps", str(10 ** 9), "--seed", str(c.seed),
        "--lr", str(c.lr), "--beta", str(c.beta), "--temperature", str(c.temperature),
        "--num-generations", str(c.num_generations),
        "--batch-size", str(c.micro_batch), "--grad-accum", str(c.grad_accum),
        "--generation-batch-size", str(max(c.generation_batch, c.num_generations)),
        "--report-to", "wandb", "--log-completions", "--sample-every", str(c.sample_every),
    ])
    trainer = train(args)

    model = trainer.model.merge_and_unload().eval()
    tok = trainer.processing_class
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    evaluate_and_log(run, model, tok, c, out)
    return 0


def rescore(run_path: str) -> int:
    """Evaluate a trial's saved adapter into its existing run, tagged `rescored`."""
    import torch
    from peft import PeftModel

    from collect_histories import load_model

    entity, project, run_id = run_path.split("/")
    run = wandb.init(entity=entity, project=project, id=run_id, resume="must")
    c = SimpleNamespace(**dict(run.config))
    out = Path(c.out_dir) / run.id
    print(f"rescoring {run.id}: {c.model} + {out}, lr={c.lr} beta={c.beta} temperature={c.temperature} "
          f"num_generations={c.num_generations}; eval {c.eval_episodes} x {c.eval_length}, "
          f"batch {c.batch_size}, seed {c.seed}", flush=True)
    base, tok = load_model(c.model)
    model = PeftModel.from_pretrained(base, str(out)).merge_and_unload().eval()
    evaluate_and_log(run, model, tok, c, out, rescored=True)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def evaluate_and_log(run, model, tok, c, out, rescored=False):
    """Run the eval loop on a trained model and log curves, summaries and the discovery plot to the run.

    rescored=True marks the run (tag, summary flag, note) once its eval has succeeded.
    """
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
                   "eval/first3_commands": r["mean_first3_commands"],
                   "eval/real_commands": r["mean_real_commands"],
                   "eval/median_name_length": r["median_name_length"],
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
        if "mean_first3_commands" in base[-1]:
            summary["base_final_first3_commands"] = base[-1]["mean_first3_commands"]
    plot(plot_rows, labels, eval_dir / "discovery_curve.png")

    with open(eval_dir / "trained_histories.jsonl", encoding="utf-8") as f:
        first = json.loads(f.readline())["history"]
    examples = wandb.Table(columns=["command", "parameters"],
                           data=[[h["command"], json.dumps(h["parameters"])] for h in first])

    wandb.log({
        **summary,
        "final_unique_commands": rows[-1]["mean_unique_commands"],
        "final_plausible_unique_commands": rows[-1]["mean_plausible_unique_commands"],
        "final_first3_commands": rows[-1]["mean_first3_commands"],
        "final_real_commands": rows[-1]["mean_real_commands"],
        "final_median_name_length": rows[-1]["median_name_length"],
        "eval_reward_mean": statistics.mean(r["mean_reward"] for r in rows),
        **{f"{t}_rate": statistics.mean(r[f"{t}_rate"] for r in rows) for t in TIERS},
        "eval/discovery_curve": wandb.Image(str(eval_dir / "discovery_curve.png")),
        "eval/example_list": examples,
    })
    if rescored:
        run.tags = sorted(set(run.tags) | {"rescored"})
        run.summary["rescored"] = True
        run.notes = ((run.notes or "") + "\nEval failed during the sweep; final_* and eval/* were computed "
                     "afterwards from the saved adapter (wandb_trial.py --rescore).").strip()
    run.finish()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--rescore"]:
        raise SystemExit(max(rescore(p) for p in sys.argv[2:]))
    raise SystemExit(main())
