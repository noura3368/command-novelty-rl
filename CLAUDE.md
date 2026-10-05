# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GRPO post-training (TRL) that teaches small instruct models to keep producing commands they have not produced before. The first target is the Korad KA3005P power supply over RS232. The model answers with one JSON object, `{"command": str, "parameters": [str, ...]}`. Each answer is rewarded only for novelty against the history shown in its prompt. Correctness is not rewarded. README.md has the reward table and the full pipeline commands.

There is no test suite, linter, or build step. The scripts are run directly with `python <script>.py` (`pip install -r requirements.txt`). Training and eval expect a CUDA GPU; `load_model` falls back to MPS or CPU.

## Pipeline and commands

```bash
python collect_histories.py --model Qwen/Qwen2.5-7B-Instruct --out histories_round0.jsonl   # build prompts
python train_grpo.py --model Qwen/Qwen2.5-7B-Instruct --histories histories_round0.jsonl --output-dir runs/round0 --lora
python evaluate.py --models Qwen/Qwen2.5-7B-Instruct runs/round0/merged --labels base round0 --steps 100 --out-dir eval/round0
python sweep.py --model Qwen/Qwen2.5-3B-Instruct --hours 20 --out-dir sweeps/qwen3b       # local random sweep, resumable
./prepare_sweep.sh Qwen/Qwen2.5-3B-Instruct sweeps/3b [batch_size]                        # inputs for the W&B sweep
wandb sweep sweep_3b.yaml && wandb agent --count 15 <entity/project/id>                  # W&B Bayesian sweep (run from repo root)
```

Sweeps run on two lab machines over SSH: `ssh sf` (conda env `novelty-command`; Qwen2.5-3B; RTX A5000s are `nvidia-smi` GPUs 1 and 2, GPU 0 is a Quadro K620 that must not be used, so set `CUDA_DEVICE_ORDER=PCI_BUS_ID`) and `ssh nkhajehn@dgx2.esg.uwaterloo.ca` (conda env `novelty-rl`; Qwen2.5-7B; one GB10). Non-interactive SSH shells do not have conda on PATH; use `~/miniconda3/envs/<env>/bin/python`. Package versions in `requirements.txt` are pinned to what those envs have.

With `--lora`, `train_grpo.py` saves the adapter in `<output_dir>` and a merged full model in `<output_dir>/merged`. The next round's `--model` is that merged folder. `--resume` continues from the latest `checkpoint-*`.

## Architecture: how the files depend on each other

- **`prompting.py` + `template.jinja`** render the single user message. The history is formatted as one JSON object per line. The docstring requires this format to stay stable so prompts stay comparable across rounds and models.
- **`reward.py`** is the single source of truth for scoring. `parse_completion` decides what counts as valid. Only a single JSON object passes, optionally wrapped in a ```json fence. `score_group` assigns tiers: `broken` / `repeat` / `new_value` / `new_structure`. A `new_structure` reward is split across the group samples that found the same new command. `REWARDS` holds the values. Three places use these functions:
  - `make_reward_fn` is the TRL reward function used in training.
  - `evaluate.py` uses them to classify each eval reply into a tier.
  - `collect_histories.py` uses `parse_completion` to decide what gets appended to a history.
- **`collect_histories.run_episodes`** is the shared exploration loop. It starts from an empty history and generates one command per step. It appends each parseable reply that is not an exact repeat. `evaluate.py` reuses it, so changing it changes both data collection and evaluation.
- **Dataset contract**: the history JSONL rows hold `history` as a **JSON string**, not a list. `train_grpo.py` passes it through as a dataset column, and TRL hands it to the reward function as the `history` kwarg.
- **Grouping assumption**: `make_reward_fn` finds GRPO groups by hashing the prompt text inside a single reward call. This only works on one GPU/process.
- **Sweeps**:
  - `sweep.py` drives the other scripts as **subprocesses via their CLI flags**, so renaming a flag in `train_grpo.py` / `evaluate.py` / `collect_histories.py` breaks it. Resume works by replaying the RNG for the trials already in `trials.csv`. Changing `RANGES` or its order changes the configs a resumed run samples.
  - `wandb_trial.py` (launched by `wandb agent` per `sweep_3b.yaml` / `sweep_7b.yaml`) imports `build_parser`/`train` from `train_grpo.py` and `evaluate_loaded`/`plot` from `evaluate.py`. It evaluates the merged model in memory. The sweep metric is `final_unique_commands`. The paths in the sweep YAML (histories, base eval CSV) are created beforehand by `prepare_sweep.sh`, whose eval settings must match the YAML's.
  - `wandb_trial.py` copies `run.config` into a namespace **before** training, because the transformers W&B callback writes all `GRPOConfig` fields into `run.config` and overwrites same-named keys (that is why the sweep key is `eval_length`, not `eval_steps`). Don't name new sweep keys after `TrainingArguments`/`GRPOConfig` fields.
- **GPU memory**: TRL loads a model given by name in fp32 unless told otherwise, so `train_grpo.py` passes `model_init_kwargs={"dtype": "bfloat16"}` on CUDA. The peak is in TRL's generation step (prefill of long history prompts, with PEFT's fp32 LoRA path), not in backprop; `--generation-batch-size` bounds it independently of `--batch-size`/`--grad-accum`, and must be divisible by `--num-generations`. sf's A5000s are shared with an `ollama` server (~4 GB per GPU).
- **W&B logging** goes through TRL 1.14's reward-function hooks. `make_reward_fn` takes `trainer_state`, `log_extra` (adds `tier`/`command`/`parameters` columns to TRL's `completions` table, enabled by `--log-completions`) and `log_metric` (the `tier/*` and `novelty/*` metrics, averaged per logging step next to `reward`/`reward_std`). With `--sample-every N` it also re-logs a growing `samples` W&B table with `commit=False`, so it attaches to the next trainer log.

## Conventions to keep consistent

- The defaults for `--target`, `--interface`, `--max-new-tokens`/`--max-completion-length` (64), and `--batch-size` are repeated in every script and in `sweep.yaml`. Change them everywhere together, or train and eval prompts will diverge.
- Evaluation always samples at temperature 1.0 with a fixed seed, so runs can be compared. The training temperature is what the sweeps tune.
- Outputs (`runs/`, `sweeps/`, `histories*.jsonl`, `wandb/`) are gitignored. `samples.jsonl` in each output dir logs every scored completion, which is where reward hacking shows up.
