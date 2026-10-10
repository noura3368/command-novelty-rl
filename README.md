# command-novelty-rl

GRPO post-training to make small language models keep generating commands they have not generated before. The first target is the Korad KA3005P power supply over RS232.

The model answers with one JSON object per response:

```json
{"command": "VSET1", "parameters": ["2.5"]}
```

The prompt lists everything already sent ("Do NOT create entries with these commands"). Each answer is scored only against that list:

| Answer | Reward (v4) |
|---|---|
| Not a single JSON object, or empty command | -10 |
| Same command words as an entry in the list (ROUGE-L 1), whatever the parameters | -6 |
| ROUGE-L 0.7 or more to an entry (near-duplicate) | -3 |
| ROUGE-L below 0.7 to every entry | +5, split across the group samples within 0.7 of each other |

Only the `command` field is judged. It is turned into words first (`command_words` in `reward.py`): cut at symbols, case changes and digits (`SOUR:VOLT`, `SetOutputVoltage`, `VSET1`), digit runs become `#`, glued words of 8+ letters are split with a fixed dictionary (`GETBATTSTATMAX` -> `GET BATT STAT MAX`), then plural S, repeated words and filler words (`THE`, `OF`, ...) are dropped and the first 6 words kept. Two names are compared by ROUGE-L: 2 x their longest common subsequence of words / their total number of words. So `Set the Output Voltages` repeats `SET OUTPUT VOLTAGE` (1.0), `SET OUTPUT VOLTAGE LEVEL` is a near-duplicate (0.86), and `SET OUTPUT CURRENT` is new (0.67).

Parameter novelty is a separate problem (values have an order, 3 < 4 < 5; commands do not) and is not rewarded. Correctness is not rewarded either, so gibberish commands count like any other. The values are in `REWARDS` and `SIM_THRESHOLD` in `reward.py`. Up to v3 the reward compared exact names: -6 broken, -1 for a known name, +5 for a new one.

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

`sweep_3b.yaml` and `sweep_7b.yaml` run a Bayesian W&B sweep over the learning rate (1e-5 to 1e-4), `beta`, training temperature and `num_generations` (4, 8 or 16), maximizing `final_unique_commands`. `wandb agent` starts `wandb_trial.py` once per trial: it trains a LoRA adapter for 150 updates, evaluates it with the same loop as `evaluate.py` (16 lists of 150 commands; the earlier 60 let both models reach the maximum of 60), and logs everything to the run. Each sweep needs its shared histories and the base model's evaluation first; `prepare_sweep.sh` builds both and skips what already exists.

```bash
wandb login                                                    # once per machine
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./prepare_sweep.sh Qwen/Qwen2.5-3B-Instruct sweeps/3b         # 7B: ./prepare_sweep.sh Qwen/Qwen2.5-7B-Instruct sweeps/7b 8
wandb sweep sweep_3b.yaml                                      # prints "wandb agent <entity>/command-novelty-rl/<id>"
wandb agent --count 15 <entity>/command-novelty-rl/<id>        # the id form, not the sweep's web URL
```

Run the agents from the repo folder, one per GPU; several agents can share one sweep id. `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes the GPU numbers match `nvidia-smi`. The optional third argument of `prepare_sweep.sh` is how many lists are generated at once; it must match `batch_size` in the YAML (16 for 3B, 8 for 7B, whose long prompts at 16 exhausted the GB10's shared memory).

Memory: with prompts of up to ~2,500 tokens, a 3B trial peaks at about 14 GB on one GPU. Each update uses 32 completions, run as 8 micro-batches of 4 (`micro_batch`, `grad_accum`) and generated `generation_batch` at a time (but at least one whole group, `num_generations`). sf's GPU 2 is shared with two ollama servers (about 10 GB), so the 3B sweep generates 8 at a time; at 16, group size 4 trials ran out of memory.

### v3

v2 showed two problems: 7B reached the 150-command maximum, and 3B often made names new by appending words to one phrase ("SET OUTPUT VOLTAGE RAMP UP DURATION WITH SLEW LIMIT ..."). v3 (`sweep_3b_v3.yaml`, `sweep_7b_v3.yaml`) changes:

- Lists keep one entry per command name, in training prompts and in the eval, matching the reward (new parameters on a known command no longer enter the list). v3 eval numbers are therefore not comparable with v2's; each v3 sweep has its own base-model eval.
- Training prompts are collected by the most diverse v2 trial (3B `35t8q5v4`, 7B `den3heif`), merged with `merge_adapter.py`, over 150 steps: with one entry per name, the base models' own lists would stay a few entries long.
- The sweep maximizes `final_first3_commands`, the number of distinct first three words of the names, which padding does not raise. Distinct names, real KA3005P commands and name length are logged next to it.
- Narrower ranges from the v2 results: lr 4e-5 to 2e-4, beta 0 to 0.03, group size 8 or 16.

The reward, prompt, training length and eval size are unchanged.

### v4

v3 models still made names "new" in ways the first-3-words metric missed: CamelCase (`SetOverloadProtectionAdjustmentFactors` counted as one word), a swapped verb in front of a copied tail, counters (`C1111`, `C1112`). v4 (`sweep_3b_v4.yaml`, `sweep_7b_v4.yaml`) changes:

- The reward compares command words by ROUGE-L (table above): repeats -6, near-duplicates -3, broken JSON -10. `sim_threshold` (0.7) is a sweep value, so it can be tuned later; the eval always uses 0.7.
- The eval list adds every reply that is not a repeat, and counts distinct commands at ROUGE-L 0.5, 0.6, 0.7, 0.8 and 0.9 (`DistinctCommands`). The sweep maximizes `final_distinct_t70`; the lower thresholds show how many commands are borderline, and the `eval/borderline_pairs` table lists them. `first3` is gone.
- Training prompts are the pooled v3 prompts of both models, deduplicated (15,204: 5,694 from 3B, 9,510 from 7B), one file for both sizes.
- The splitting dictionary, `command_words.txt.gz`, was built once from those prompts with `build_dictionary.py` (656 of the models' own words, then wordninja's English list). It is part of the metric: do not rebuild it during a study.
- Search ranges are v2's: lr 1e-5 to 1e-4, beta 0 to 0.1, temperature 0.8 to 1.3, group size 4, 8 or 16. 10 trials per model, run on Nibi (`compute_canada/agent.sh`).

What each run logs:

| Where | What |
|---|---|
| `train/reward`, `train/reward_std` | mean and standard deviation of the reward over each step's completions |
| `train/tier/*_rate` | share of broken / repeat / near_duplicate / new_structure answers (v1-v3: new_value instead of near_duplicate) |
| `train/novelty/distinct_new_commands` (`_per_group`) | different new commands in the step (averaged per prompt); falling while `new_structure_rate` stays high means the model found one "new" command for every prompt |
| `train/novelty/nonstandard_new_command_rate` | share of new commands that do not look like `VSET1:` / `*IDN?` style syntax (`plausible_syntax` in `reward.py`), a rough gibberish signal |
| `train/frac_reward_zero_std` | share of prompts whose samples all scored the same, which give GRPO nothing to learn from |
| `train/kl`, `train/entropy`, `train/completions/*` | drift from the base model, sampling diversity, answer length |
| `completions` table | every completion of every step with its prompt, tier, parsed command, reward and advantage |
| `samples` table | two whole groups every 10 steps with the end of their history, growing over the run |
| `eval/*` | per eval step: distinct commands at each threshold (`eval/distinct_t50` ... `t90`, v4), unique commands, plausible unique commands, reward mean and std, tier rates |
| `eval/borderline_pairs` table | v4: commands new at 0.7 but not at 0.5, next to the earlier command they nearly match |
| summary | `final_distinct_t70` (v4 sweep metric) and `final_distinct_t50` ... `t90`, `final_unique_commands` (v2 sweep metric), `final_first3_commands` (v3 sweep metric, v3 runs only), `final_real_commands` (names in the documented KA3005P set), `final_median_name_length`, `final_plausible_unique_commands`, `gain_over_base` (at 0.7 in v4), mean eval reward and tier rates, the discovery curve against the base model |

The eval reward scores each reply as a group of one, so a new command always gets the full +5 there.

A checkpoint is saved every 100 steps (`--save-steps`). If training stops, rerun the same command with `--resume` to continue from the latest checkpoint.

`--lora` trains a small adapter instead of all weights. Use it on a 24GB GPU for any size: training all weights of a 1.5B model already needs about 24GB. The reward groups samples by prompt, which assumes a single GPU.
