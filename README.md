# command-novelty-rl

GRPO post-training to make small language models keep generating commands they have not generated before. The first target is the Korad KA3005P power supply over RS232.

The model answers with one JSON object per response:

```json
{"command": "VSET1", "parameters": ["2.5"]}
```

The prompt lists everything already sent ("Do NOT create entries with these commands"). Each answer is scored only against that list:

| Answer | Reward |
|---|---|
| Not a single JSON object, or empty command | -6 |
| Same command and parameters as an entry in the list | -1 |
| Command in the list, new parameters | +0.5 |
| Command not in the list | +5, split across the group samples that found the same one |

Correctness is not rewarded, so gibberish commands count like any other. The values are in `REWARDS` in `reward.py`.

## Pipeline

1. **Build prompts.** `collect_histories.py` lets a model explore from an empty list, one command at a time, adding each new command to the list. The list before every step is saved as one training prompt.
2. **Train.** `train_grpo.py` samples `--num-generations` answers per prompt, scores them, and reinforces the ones above the group mean. Every scored answer is logged to `<output_dir>/samples.jsonl`.
3. **Repeat.** Re-collect prompts with the trained model and train again.

```bash
pip install -r requirements.txt

python collect_histories.py --model Qwen/Qwen2.5-7B-Instruct --out histories_round0.jsonl
python train_grpo.py --model Qwen/Qwen2.5-7B-Instruct --histories histories_round0.jsonl \
    --output-dir runs/round0 --lora

# next round: use the merged model
python collect_histories.py --model runs/round0/merged --out histories_round1.jsonl
python train_grpo.py --model runs/round0/merged --histories histories_round1.jsonl \
    --output-dir runs/round1 --lora
```

## Evaluation

`evaluate.py` runs each model through the same one-command loop (same prompt, settings and seed) and records, per step, the mean number of unique commands per list, the list length, and the share of broken / repeat / new-value / new-command answers. It writes `eval_steps.csv`, each model's final lists, and `discovery_curve.png`.

```bash
python evaluate.py --models Qwen/Qwen2.5-7B-Instruct runs/round0/merged --labels base round0 \
    --steps 100 --out-dir eval/round0
```

## Hyperparameter sweep

`sweep.py` builds one shared histories file and evaluates the base model, then keeps running random trials until `--hours` is used up. Each trial samples a learning rate (1e-6 to 1e-4, log scale), `beta` (0 to 0.1) and training temperature (0.8 to 1.3), trains a LoRA adapter for `--trial-steps` updates, evaluates it, and appends a row to `trials.csv`. Rerunning the same command continues after the last finished trial.

```bash
python sweep.py --model Qwen/Qwen2.5-3B-Instruct --hours 20 --out-dir sweeps/qwen3b
```

### W&B sweep

`sweep_3b.yaml` and `sweep_7b.yaml` run a Bayesian W&B sweep over the learning rate, `beta`, training temperature and `num_generations` (4, 8 or 16), maximizing `final_unique_commands`. `wandb agent` starts `wandb_trial.py` once per trial: it trains a LoRA adapter for 150 updates, evaluates it with the same loop as `evaluate.py`, and logs everything to the run. Each sweep needs its shared histories and the base model's evaluation first; `prepare_sweep.sh` builds both and skips what already exists.

```bash
wandb login                                                    # once per machine
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./prepare_sweep.sh Qwen/Qwen2.5-3B-Instruct sweeps/3b
wandb sweep sweep_3b.yaml                                      # prints <entity/project/id>
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 wandb agent --count 30 <entity/project/id>
```

Run the agents from the repo folder, one per GPU; several agents can share one sweep id. `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes the GPU numbers match `nvidia-smi`.

What each run logs:

| Where | What |
|---|---|
| `train/reward`, `train/reward_std` | mean and standard deviation of the reward over each step's completions |
| `train/tier/*_rate` | share of broken / repeat / new_value / new_structure answers |
| `train/novelty/distinct_new_commands` (`_per_group`) | different new commands in the step (averaged per prompt); falling while `new_structure_rate` stays high means the model found one "new" command for every prompt |
| `train/novelty/nonstandard_new_command_rate` | share of new commands that do not look like `VSET1:` / `*IDN?` style syntax (`plausible_syntax` in `reward.py`), a rough gibberish signal |
| `train/frac_reward_zero_std` | share of prompts whose samples all scored the same, which give GRPO nothing to learn from |
| `train/kl`, `train/entropy`, `train/completions/*` | drift from the base model, sampling diversity, answer length |
| `completions` table | every completion of every step with its prompt, tier, parsed command, reward and advantage |
| `samples` table | two whole groups every 10 steps with the end of their history, growing over the run |
| `eval/*` | per eval step: unique commands, plausible unique commands, reward mean and std, tier rates |
| summary | `final_unique_commands` (the sweep metric), `final_plausible_unique_commands`, `gain_over_base`, mean eval reward and tier rates, the discovery curve against the base model |

The eval reward scores each reply as a group of one, so a new command always gets the full +5 there.

A checkpoint is saved every 100 steps (`--save-steps`). If training stops, rerun the same command with `--resume` to continue from the latest checkpoint.

`--lora` trains a small adapter instead of all weights. Use it on a 24GB GPU for any size: training all weights of a 1.5B model already needs about 24GB. The reward groups samples by prompt, which assumes a single GPU.
