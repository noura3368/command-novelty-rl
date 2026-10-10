#!/usr/bin/env python3
"""
Build command_words.txt.gz, the dictionary reward.command_words uses to split glued names
(GETBATTSTATMAX -> GET BATT STAT MAX).

The file lists words most likely first:
  1. the models' own words: every word that appears, cut out by its own spelling (spaces,
     symbols, CamelCase), in at least --min-names distinct command names of the training
     histories (SET, VOLTAGE, BATT, OVP, ...), most common first;
  2. then wordninja's English word list (126k words ranked by frequency, MIT licence), for
     words the models never wrote separately.
A header line '#model_words <n>' marks how many words come from part 1.

Build it once from the training prompts and commit it: the reward and every eval read this
file, so rebuilding it changes the metric. It needs `pip install wordninja`, only here.

    python build_dictionary.py --histories sweeps/3b-v4/histories.jsonl
"""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from reward import DICTIONARY, explicit_pieces


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--histories", nargs="+", required=True, help="training-prompt JSONL files (history as JSON string)")
    ap.add_argument("--min-names", type=int, default=3, help="keep words seen in at least this many distinct names")
    ap.add_argument("--out", default=str(DICTIONARY))
    args = ap.parse_args()

    import wordninja
    english_file = Path(wordninja.__file__).parent / "wordninja" / "wordninja_words.txt.gz"
    with gzip.open(english_file, "rt", encoding="utf-8") as f:
        english = [w for w in f.read().split() if w.isalpha()]

    names = set()
    for path in args.histories:
        with open(path, encoding="utf-8") as f:
            for line in f:
                names.update(x["command"] for x in json.loads(json.loads(line)["history"]))

    counts = Counter()
    for name in names:
        pieces = [p.lower() for p in explicit_pieces(name) if p.isalpha()]
        if len(pieces) > 1:   # only names whose spelling marks their word breaks
            counts.update({p for p in pieces if 2 <= len(p) <= 12})
    model_words = [w for w, c in counts.most_common() if c >= args.min_names]
    seen = set(model_words)
    words = model_words + [w for w in english if w not in seen]

    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        f.write(f"#model_words {len(model_words)}\n" + "\n".join(words))
    print(f"{len(names)} distinct names -> {len(model_words)} model words (e.g. {', '.join(model_words[:12])}) "
          f"+ {len(words) - len(model_words)} English words -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
