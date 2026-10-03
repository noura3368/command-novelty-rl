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

A checkpoint is saved every 100 steps (`--save-steps`). If training stops, rerun the same command with `--resume` to continue from the latest checkpoint.

`--lora` trains a small adapter instead of all weights. Use it on a 24GB GPU for any size: training all weights of a 1.5B model already needs about 24GB. The reward groups samples by prompt, which assumes a single GPU.
