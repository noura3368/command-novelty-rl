#!/usr/bin/env python3
"""
GRPO post-training for command novelty (TRL).

Every row of the histories file becomes one prompt. For each prompt GRPO samples
--num-generations single commands and reinforces those that score above the
group mean under reward.py. Every scored completion is logged to
<output_dir>/samples.jsonl so loopholes show up early.

    python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --histories histories_round0.jsonl \
        --output-dir runs/round0

For bigger models add --lora: only a small adapter is trained, and the KL reference is the
base model with the adapter switched off, so no second copy is kept in memory. The adapter is
saved in <output_dir> and a merged full model in <output_dir>/merged; pass the merged folder
as --model to collect_histories.py and to the next round.

A checkpoint is saved every --save-steps steps. If a run stops, rerun the same command
with --resume to continue from the latest one.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from prompting import build_messages
from reward import make_reward_fn


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local checkpoint")
    ap.add_argument("--histories", nargs="+", required=True, help="JSONL files from collect_histories.py")
    ap.add_argument("--target", default="KORAD KA3005P Power Supply")
    ap.add_argument("--interface", default="RS232 interface")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-generations", type=int, default=8, help="samples per prompt (GRPO group size)")
    ap.add_argument("--max-completion-length", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-6, or 1e-5 with --lora")
    ap.add_argument("--beta", type=float, default=0.04, help="KL penalty to the base model")
    ap.add_argument("--batch-size", type=int, default=8, help="completions per device per step")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N updates (overrides --epochs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=100, help="save a checkpoint every N steps")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the latest checkpoint in --output-dir (use the same settings as before)")
    ap.add_argument("--use-vllm", action="store_true", help="generate with vLLM (much faster, needs vllm installed)")
    ap.add_argument("--report-to", default="none", help="e.g. wandb")
    ap.add_argument("--lora", action="store_true", help="train a LoRA adapter instead of all weights")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    return ap


def train(args) -> GRPOTrainer:
    """Train on the histories and save the result (the adapter, with --lora) in --output-dir."""
    rows = []
    for path in args.histories:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append({"prompt": build_messages(args.target, args.interface, json.loads(r["history"])),
                             "history": r["history"]})
    dataset = Dataset.from_list(rows).shuffle(seed=args.seed)
    print(f"{len(rows)} prompts from {len(args.histories)} file(s)")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = GRPOConfig(
        output_dir=str(out),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        learning_rate=args.lr if args.lr is not None else (1e-5 if args.lora else 1e-6),
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        seed=args.seed,
        logging_steps=1,
        save_steps=args.save_steps,
        bf16=torch.cuda.is_available(),
        use_vllm=args.use_vllm,
        report_to=args.report_to,
    )
    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules="all-linear",
                                 task_type="CAUSAL_LM")
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=make_reward_fn(out / "samples.jsonl"),
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    has_checkpoint = any(out.glob("checkpoint-*"))
    if args.resume and not has_checkpoint:
        print(f"--resume: no checkpoint in {out}, starting from step 0")
    trainer.train(resume_from_checkpoint=args.resume and has_checkpoint)
    trainer.save_model(str(out))
    return trainer


def main() -> int:
    args = build_parser().parse_args()
    trainer = train(args)
    out = Path(args.output_dir)
    if args.lora:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(out / "merged"))
        trainer.processing_class.save_pretrained(str(out / "merged"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
