#!/usr/bin/env python3
"""
Remove reward-hacked names from the training prompts' lists (histories.jsonl), so the model never
sees them presented as valid commands. Two rules, applied to each list in order:

  1. wordy:  the name has MIN_CONNECTORS or more CONNECTORS words (reward.wordy), i.e. padding or a
             sentence: SetOutputCurrentTo100mAAndVerifyResponse, Transfer Settings from Flash to RAM
  2. repeat: the name matches an earlier name in the list once case, separators and numbers are ignored
             (same_name): LOAD SET 5A / LOAD SET 2A, BOOST_VOLTAGE_TEST_1 / _3, CURR_MAX / CURRMAX.
             The first spelling is kept; a number in a name is a parameter written into it.

Removed names do not count as earlier names for rule 2. Each row keeps its episode, step and source
fields; rows whose cleaned list equals an earlier row's are dropped.

    python clean_histories.py sweeps/pooled/histories.jsonl --out sweeps/pooled/histories_clean.jsonl
    python clean_histories.py sweeps/pooled/histories.jsonl --out clean.jsonl --removed removed.csv
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from reward import wordy


def same_name(name: str) -> str:
    """The name with case, separators and numbers ignored: 'Set_Output 12V' and 'SETOUTPUT5V' -> 'SETOUTPUT#V'."""
    return re.sub(r"\d+", "#", re.sub(r"[^A-Z0-9]", "", name.upper()))


def clean(history: list) -> tuple:
    """(kept entries, [(rule, name, earlier name or ''), ...] for the removed ones)."""
    kept, removed, seen = [], [], {}
    for x in history:
        n = x["command"]
        k = same_name(n)
        if wordy(n):
            removed.append(("wordy", n, ""))
        elif k in seen:
            removed.append(("repeat", n, seen[k]))
        else:
            seen[k] = n
            kept.append(x)
    return kept, removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("histories", type=Path, help="training prompts (history as a JSON string)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--removed", type=Path, help="write every distinct removed name with its rule to this CSV")
    args = ap.parse_args()
    if args.out.resolve() == args.histories.resolve():
        ap.error("--out must be a different file")

    rows_in = entries_in = changed = duplicates = 0
    removed_counts, removed_names = Counter(), Counter()
    seen_lists, out_rows = set(), []
    with open(args.histories, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            h = json.loads(r["history"])
            rows_in += 1
            entries_in += len(h)
            kept, removed = clean(h)
            changed += bool(removed)
            removed_counts.update(rule for rule, _, _ in removed)
            removed_names.update(removed)
            text = json.dumps(kept)
            if text in seen_lists:
                duplicates += 1
                continue
            seen_lists.add(text)
            out_rows.append({**r, "history": text})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"{rows_in} prompts, {entries_in} list entries")
    for rule in ("wordy", "repeat"):
        print(f"  {rule:6s} entries removed: {removed_counts[rule]} ({100 * removed_counts[rule] / entries_in:.2f}%), "
              f"{len({n for (ru, n, _) in removed_names if ru == rule})} distinct names")
    print(f"  prompts changed: {changed} ({100 * changed / rows_in:.0f}%)")
    print(f"  duplicate prompts dropped after cleaning: {duplicates}")
    print(f"{len(out_rows)} prompts -> {args.out}")

    if args.removed:
        with open(args.removed, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rule", "name", "earlier_name", "times_removed"])
            w.writerows([*k, c] for k, c in sorted(removed_names.items(), key=lambda kc: (kc[0][0], -kc[1])))
        print(f"{len(removed_names)} distinct removals -> {args.removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
