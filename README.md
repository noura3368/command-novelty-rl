# command-novelty-rl

GRPO post-training to make small language models keep generating commands they have not generated before. The first target is the Korad KA3005P power supply over RS232.

The model answers with one JSON object per response:

```json
{"command": "VSET1", "parameters": ["2.5"]}
```

The prompt lists everything already sent ("Do NOT create entries with these commands"). Each answer is scored only against that list:

| Answer | Reward (v5) |
|---|---|
| Not a single JSON object, or empty command | -10 |
| Exactly the name of an entry in the list, whatever the parameters | -6 |
| Any other name (new) | +5, split across the group samples with the same name (as in v3) |

Only the `command` field is judged, and since v5 exactly as written: `SET_VOLTAGE` after `SetVoltage`, `TEST2` after `TEST1` or a padded sentence all earn +5. The v5 reward detects no hacks on purpose; the v5 sweep measures which settings lead to them (see "v5").

v4's reward (`sweep_*_v4.yaml`) compared command words instead and had a fourth tier: -10 broken, -6 wordy (2 or more connector words such as `WITH`, `AND`, `TO`: reads like a sentence, not a command), -6 same command words as an entry, +5 split across the group samples with the same command words. Since v5 these functions only measure hacking (`clean_histories.py`). In v4 the name was turned into words first (`command_words` in `reward.py`): cut at symbols, case changes and digits (`SOUR:VOLT`, `SetOutputVoltage`, `VSET1`), digit runs become `#`, glued words of 8+ letters are split with a fixed dictionary (`GETBATTSTATMAX` -> `GET BATT STAT MAX`), then plural S, repeated words and filler words (`THE`, `OF`, ...) are dropped and the first 6 words kept. Two names with the same words are the same command. So `Set the Output Voltages` and `set_output_voltage` repeat `SET OUTPUT VOLTAGE`, `VSET2` repeats `VSET1` (`VSET #`), and `SET OUTPUT VOLTAGE LEVEL` and `SET OUTPUT CURRENT` are new.

Before that, a name with 2 or more connector words (`CONNECTORS` and `wordy` in `reward.py`, counted on the same words before filler is dropped) is `wordy`: `SET BOOST MODE TO BOOSTBOOST WITH OVERRIDE AND CUSTOM LEVEL`, `GETBATTERYCHARGESTATUSWITHTIMEANDTEMPERATURE`. In v2 these made up up to 68% of a 3B trial's names and 91% of a 7B trial's (`hack_report.py`). Real commands (`VSET1:5`, `SOUR:VOLT:LEVE 5`) have none, and one is allowed (`SET OUTPUT VOLTAGE TO 5`). `A` and `ON` are not connectors, since they are also units and values (`CURR 1 A`, `OUTP ON`).

Parameter novelty is a separate problem (values have an order, 3 < 4 < 5; commands do not) and is not rewarded. Correctness is not rewarded either, so gibberish commands count like any other. The values are in `REWARDS` in `reward.py`. Up to v3 the reward compared exact names: -6 broken, -1 for a known name, +5 for a new one.

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

- The reward compares command words (table above): broken JSON -10, `wordy` (2 or more connector words) -6, repeats -6, new +5. There is no similarity threshold: an earlier v4 draft also had a ROUGE-L near-duplicate tier (-3), removed before any v4 run.
- The eval list adds every reply that is neither broken, wordy nor a repeat, so the model never sees its wordy names in later prompts. The sweep maximizes `final_clean_commands`: the number of entries per list, i.e. the replies the reward scored as new. `first3` is gone.
- Training prompts are the pooled v3 prompts of both models, deduplicated (15,204: 5,694 from 3B, 9,510 from 7B), one file for both sizes.
- The splitting dictionary, `command_words.txt.gz`, was built once from those prompts with `build_dictionary.py` (656 of the models' own words, then wordninja's English list). It is part of the metric: do not rebuild it during a study.
- Search ranges are v2's: lr 1e-5 to 1e-4, beta 0 to 0.1, temperature 0.8 to 1.3, group size 4, 8 or 16. 10 trials per model, run on Nibi (`compute_canada/agent.sh`).

### v5 (branch `v5-simple-reward`)

v4 kept adding rules against each new hack. v5 asks a different question: with a simple reward that detects no hacks, which settings lead models to hack? `sweep_0.5b_v5.yaml`, `sweep_3b_v5.yaml`, `sweep_7b_v5.yaml` change:

- The reward compares exact names (table above): broken JSON -10, a name already in the list -6, a new name +5 split across the group samples with exactly that name (as in v3).
- Eval lists add every reply whose exact name is new (`evaluate.py --list-rule exact`, the default), 30 lists per trial. Each trial logs three numbers: `final_new_commands`, the list length the reward pays for (the sweep metric); `final_clean_commands`, the same lists recounted without hacks by `clean_histories.py`'s rules (no wordy names, no repeats once case, separators and numbers are ignored); and `final_hack_share`, the share of the lists those rules remove.
- Training prompts are v4's pooled prompts after `clean_histories.py` removed wordy names and those repeats from every list, then the duplicate prompts this left (14,732 of 15,204): `sweeps/pooled/histories_clean.jsonl`.
- Random search instead of Bayesian, 30 trials per model, so trials spread evenly over v2's ranges instead of gathering where the metric is high (which is where hacking pays).
- The baseline is the untrained models with the same eval, seeds 0, 1 and 2 (`sweeps/baseline/<size>/seed<N>/`); `base_eval` is the seed-0 run, the trials' own seed.

```bash
python clean_histories.py histories.jsonl --out sweeps/pooled/histories_clean.jsonl --removed sweeps/pooled/removed.csv
python evaluate.py --models Qwen/Qwen2.5-3B-Instruct --labels base --episodes 30 --steps 150 --seed 0 \
    --out-dir sweeps/baseline/3b/seed0                  # once per size and seed
wandb sweep sweep_3b_v5.yaml                            # then compute_canada/agent.sh, one trial per job
```

What each run logs:

| Where | What |
|---|---|
| `train/reward`, `train/reward_std` | mean and standard deviation of the reward over each step's completions |
| `train/tier/*_rate` | share of broken / repeat / new_structure answers (v4 also wordy; v1-v3 new_value) |
| `train/rewardspy/*` | rewardspy's detector states (0 OK, 1 warning, 2 alert) on the reward numbers; its log is `rewardspy.jsonl` in the trial folder |
| `train/novelty/distinct_new_commands` (`_per_group`) | different new commands in the step (averaged per prompt); falling while `new_structure_rate` stays high means the model found one "new" command for every prompt |
| `train/novelty/nonstandard_new_command_rate` | share of new commands that do not look like `VSET1:` / `*IDN?` style syntax (`plausible_syntax` in `reward.py`), a rough gibberish signal |
| `train/frac_reward_zero_std` | share of prompts whose samples all scored the same, which give GRPO nothing to learn from |
| `train/kl`, `train/entropy`, `train/completions/*` | drift from the base model, sampling diversity, answer length |
| `completions` table | every completion of every step with its prompt, tier, parsed command, reward and advantage |
| `samples` table | two whole groups every 10 steps with the end of their history, growing over the run |
| `eval/*` | per eval step: new commands, clean commands and hack share (v5; in v4 `eval/clean_commands` was the list length), unique commands, plausible unique commands, reward mean and std, tier rates |
| summary | `final_new_commands` (v5 sweep metric), `final_clean_commands` and `final_hack_share` (v5; in v4 `final_clean_commands` was the list length and the sweep metric), `base_final_*`, `gain_over_base` and `clean_gain_over_base` (v5), `final_unique_commands` (v2 sweep metric), `final_first3_commands` (v3 sweep metric, v3 runs only), `final_real_commands` (names in the documented KA3005P set), `final_median_name_length`, `final_plausible_unique_commands`, `gain_over_base` (in clean commands in v4), mean eval reward and tier rates, the discovery curve against the base model |

The eval reward scores each reply as a group of one, so a new command always gets the full +5 there.

A checkpoint is saved every 100 steps (`--save-steps`). If training stops, rerun the same command with `--resume` to continue from the latest checkpoint.

`--lora` trains a small adapter instead of all weights. Use it on a 24GB GPU for any size: training all weights of a 1.5B model already needs about 24GB. The reward groups samples by prompt, which assumes a single GPU.
