#!/usr/bin/env python3
"""
Merge a saved LoRA adapter (e.g. a sweep trial's <out_dir>/<run id>) into its base model and
save the full model, so scripts that take a model path (collect_histories.py --model) can use it.

    python merge_adapter.py sweeps/7b-v2/wandb/den3heif sweeps/7b-v3/collector
"""

import argparse
import json
from pathlib import Path

from peft import PeftModel

from collect_histories import load_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter", help="folder with adapter_config.json")
    ap.add_argument("out", help="folder for the merged model")
    args = ap.parse_args()
    base = json.loads((Path(args.adapter) / "adapter_config.json").read_text())["base_model_name_or_path"]
    model, tok = load_model(base)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"merged {args.adapter} into {base} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
