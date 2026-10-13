#!/usr/bin/env python3
"""
Per-trial share of names that do not look like a command, the clean distinct count left after
removing them, and the trial's hyperparameters, from saved eval lists (*_histories.jsonl).

A name is flagged as a reward hack if it is reward.wordy: MIN_CONNECTORS or more CONNECTORS words,
counted after cutting it into words like reward.command_words (spaces, symbols, case changes, glued words
split with the dictionary: GETBATTERYCHARGESTATUSWITHTIME -> GET BATTERY ... WITH TIME). Real commands
(VSET1:5, ISET1?, SOUR:VOLT:LEVE 5) have none; padded names and sentences have several:
'SET BOOST MODE TO BOOSTBOOST WITH OVERRIDE AND CUSTOM LEVEL ...'. Each name is judged on its own.

Per list: names = distinct names, clean = the names v4's list rule would keep (classify: neither wordy
nor a repeat by command words), i.e. the list length a v4 eval would report. Both are averaged
over the lists in the file; flagged = flagged names / names. Within each sweep, trials are sorted
by clean, and the first is the best trial. Hyperparameters come from <trial dir>/config.yaml or
wandb/run-*-<run id>/files/config.yaml under the current folder.

    python hack_report.py sweeps/3b-v2/base_eval/base_histories.jsonl sweeps/3b-v2/wandb/*/eval/trained_histories.jsonl
    python hack_report.py sweeps/*-v2/base_eval/*_histories.jsonl sweeps/*-v2/wandb/*/eval/trained_histories.jsonl \\
        --csv hacks_v2.csv --sample hack_sample.csv

--sample writes SAMPLE_SIZE random flagged and SAMPLE_SIZE random unflagged names, with an empty
`your_label` column, to check the rule by eye.
"""

import argparse
import csv
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from reward import CONNECTORS, MIN_CONNECTORS, _split_words, command_words, wordy

HYPERPARAMS = ("lr", "beta", "temperature", "num_generations")
SAMPLE_SIZE = 50


def connectors(name: str) -> int:
    return sum(w in CONNECTORS for w in _split_words(name))


def classify(names: List[str]) -> List[str]:
    """The tier of each distinct name in list order under v4's rules, judged against the names before it:
    wordy, repeat (same command words as an earlier name) or new. The new ones are what a v4 eval list
    would hold (collect_histories.run_episodes), so their count is v4's clean commands."""
    seen, tiers = set(), []
    for n in names:
        w = command_words(n)
        tiers.append("wordy" if wordy(n) else "repeat" if w in seen else "new")
        seen.add(w)
    return tiers


def sweep_and_run(path: Path):
    """('3b-v2', '<run id>') for a trial's eval file, ('3b-v2', 'base') for the base model's."""
    sweep = next((p for p in reversed(path.parts) if re.search(r"-v\d+$", p)), path.parent.name)
    run = path.parent.parent.name if path.parent.name == "eval" else "base"
    return sweep, run


def hyperparams(path: Path) -> Dict[str, str]:
    """The sweep's searched values for a trial's eval file, from its W&B config.yaml; {} for a base eval or if missing."""
    if path.parent.name != "eval":
        return {}
    trial = path.parent.parent
    configs = [trial / "config.yaml", *sorted(Path("wandb").glob(f"run-*-{trial.name}/files/config.yaml"))]
    config = next((c for c in configs if c.exists()), None)
    if config is None:
        return {}
    text = config.read_text(encoding="utf-8")
    found = {k: re.search(rf"^{k}:\n\s+value: (.*)$", text, re.M) for k in HYPERPARAMS}
    return {k: m.group(1) for k, m in found.items() if m}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="*_histories.jsonl from evaluate.py or a sweep trial's eval/")
    ap.add_argument("--csv", type=Path, help="write the per-trial table")
    ap.add_argument("--sample", type=Path, help="write random flagged and unflagged names to check the rule")
    args = ap.parse_args()

    by_sweep, pool = defaultdict(list), {True: [], False: []}
    for f in args.files:
        sweep, run = sweep_and_run(f)
        with open(f, encoding="utf-8") as fh:
            lists: List[List[str]] = [list(dict.fromkeys(x["command"] for x in json.loads(line)["history"]))
                                      for line in fh]
        names, clean, n_flagged = [], [], 0
        for ns in lists:
            flags = [wordy(n) for n in ns]
            n_flagged += sum(flags)
            names.append(len(ns))
            clean.append(classify(ns).count("new"))
            for n, fl in zip(ns, flags):
                pool[fl].append((f"{sweep}/{run}", n))
        h = hyperparams(f)
        by_sweep[sweep].append({
            "sweep": sweep, "trial": run,
            "names": round(statistics.mean(names), 1),
            "flagged_share": round(n_flagged / sum(names), 3) if sum(names) else 0.0,
            "clean": round(statistics.mean(clean), 1),
            **{k: h.get(k) for k in HYPERPARAMS},
        })

    table = []
    for sweep in sorted(by_sweep):
        rows = sorted(by_sweep[sweep], key=lambda r: (r["trial"] != "base", -r["clean"]))
        best = next((r for r in rows if r["trial"] != "base"), None)
        print(f"\n== {sweep}: flagged = {MIN_CONNECTORS}+ connector words; sorted by clean distinct names")
        print(f"   {'trial':10s} {'names':>6s} {'flagged':>8s} {'clean':>6s} {'lr':>8s} {'beta':>6s} {'temp':>5s} {'group':>5s}")
        for r in rows:
            hp = (f"{float(r['lr']):8.1e} {float(r['beta']):6.3f} {float(r['temperature']):5.2f} {r['num_generations']:>5s}"
                  if r["lr"] else f"{'-':>8s} {'-':>6s} {'-':>5s} {'-':>5s}")
            mark = "*" if r is best else " "
            print(f" {mark} {r['trial']:10s} {r['names']:6.1f} {100 * r['flagged_share']:7.0f}% {r['clean']:6.1f} {hp}")
        table += rows
    print("\n* best trial (most clean distinct names)")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        print(f"wrote {len(table)} rows to {args.csv}")
    if args.sample:
        rng = random.Random(0)
        sample = [(fl, *x) for fl in (True, False) for x in rng.sample(pool[fl], min(SAMPLE_SIZE, len(pool[fl])))]
        rng.shuffle(sample)
        with open(args.sample, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["trial", "name", "connectors", "flagged", "your_label"])
            w.writerows([t, n, connectors(n), int(fl), ""] for fl, t, n in sample)
        print(f"wrote {len(sample)} names to {args.sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
