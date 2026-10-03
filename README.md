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

A checkpoint is saved every 100 steps (`--save-steps`). If training stops, rerun the same command with `--resume` to continue from the latest checkpoint.

`--lora` trains a small adapter instead of all weights. Use it on a 24GB GPU for any size: training all weights of a 1.5B model already needs about 24GB. The reward groups samples by prompt, which assumes a single GPU.
