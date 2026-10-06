#!/usr/bin/env python3
"""
Build GRPO training prompts by letting a model explore on its own.

Each episode starts with an empty history. At every step the model generates one
command for the current history; a parseable command that is not an exact repeat
is appended. The history *before* each step is written as one row, so the
training set covers histories of every length the model actually reaches.

Run with the base model first, then again with a trained checkpoint (--model
<output_dir>) so the histories keep up with the policy.

    python collect_histories.py --model Qwen/Qwen2.5-1.5B-Instruct --out histories_round0.jsonl
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompting import build_messages
from reward import parse_completion


def load_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    return model, tok


def run_episodes(model, tok, target, interface, episodes, steps, temperature=1.0, max_new_tokens=64,
                 batch_size=16):
    """Explore from empty histories, one command per episode per step.

    Yields (step, before, replies, after) once per step: the histories before the step,
    the raw replies, and the histories after each parseable reply that is not an exact
    repeat of its episode's history has been appended.
    """
    histories = [[] for _ in range(episodes)]
    for step in range(steps):
        before = [list(h) for h in histories]
        texts = [tok.apply_chat_template(build_messages(target, interface, h),
                                         tokenize=False, add_generation_prompt=True) for h in histories]
        replies = []
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**batch, do_sample=True, temperature=temperature,
                                     max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            replies += tok.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)

        for h, reply in zip(histories, replies):
            p = parse_completion(reply)
            if p is None or any(x["command"] == p[0] and x["parameters"] == p[1] for x in h):
                continue
            h.append({"command": p[0], "parameters": p[1]})
        # Prompts grow every step, so cached blocks from earlier steps rarely fit later ones. On a GB10,
        # whose GPU shares system memory, the cache grew until dgx2 ran out of memory around step 100.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield step, before, replies, histories


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local checkpoint")
    ap.add_argument("--target", default="KORAD KA3005P Power Supply")
    ap.add_argument("--interface", default="RS232 interface")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--steps", type=int, default=30, help="commands generated per episode")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16, help="episodes generated at once on the GPU")
    ap.add_argument("--out", required=True, help="output JSONL")
    args = ap.parse_args()

    model, tok = load_model(args.model)
    with open(args.out, "w", encoding="utf-8") as out:
        for step, before, _, after in run_episodes(model, tok, args.target, args.interface, args.episodes,
                                                   args.steps, args.temperature, args.max_new_tokens,
                                                   args.batch_size):
            for e, h in enumerate(before):
                out.write(json.dumps({"episode": e, "step": step, "history": json.dumps(h)}) + "\n")
            added = sum(len(a) > len(b) for a, b in zip(after, before))
            print(f"step {step + 1}/{args.steps}: {added}/{args.episodes} episodes added a command", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
