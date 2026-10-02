#!/usr/bin/env python3
"""
Novelty reward for GRPO with one command per response.

The model answers with one JSON object: {"command": str, "parameters": [str, ...]}.
The `command` field is the command's structure and the parameters are its values,
so no canonicalization rules are needed. Each completion is scored against the
history shown in its own prompt:

    not one JSON object / empty command      broken          -6
    same command and same parameters seen    repeat          -1
    command seen, parameters new             new_value       +0.5
    command never seen                       new_structure   +5 / k

k is the number of completions in the same group (same prompt) that produced
the same new command, so a group that all finds the same thing shares one bonus.
Gibberish commands count like any other: only novelty is rewarded.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

REWARDS = {"broken": -6.0, "repeat": -1.0, "new_value": 0.5, "new_structure": 5.0}

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_completion(text: str) -> Optional[Tuple[str, List[str]]]:
    """Return (command, parameters), or None unless the text is a single JSON object with a non-empty command.

    A surrounding ```json fence is allowed; any other text around the object is not.
    """
    text = _FENCE.sub("", text.strip()).strip()
    try:
        obj = json.loads(text)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    cmd = obj.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return None
    params = obj.get("parameters")
    if params is None:
        params = []
    elif not isinstance(params, list):
        params = [params]
    return cmd.strip(), [str(p).strip() for p in params]


def score_group(texts: List[str], history: List[dict]) -> List[Tuple[str, float, Optional[tuple]]]:
    """Score the completions of one prompt. Returns (tier, reward, parsed) per completion, in order."""
    seen_exact = {(h["command"], tuple(h["parameters"])) for h in history}
    seen_struct = {h["command"] for h in history}
    parsed = [parse_completion(t) for t in texts]

    tiers = []
    for p in parsed:
        if p is None:
            tiers.append("broken")
        elif (p[0], tuple(p[1])) in seen_exact:
            tiers.append("repeat")
        elif p[0] in seen_struct:
            tiers.append("new_value")
        else:
            tiers.append("new_structure")

    new_counts = Counter(p[0] for p, t in zip(parsed, tiers) if t == "new_structure")
    out = []
    for p, t in zip(parsed, tiers):
        r = REWARDS[t] / new_counts[p[0]] if t == "new_structure" else REWARDS[t]
        out.append((t, r, p))
    return out


def make_reward_fn(log_path: Optional[Path] = None):
    """Build the TRL reward function. Every scored completion is appended to `log_path` (JSONL) if given.

    The dataset must have a `history` column holding the prompt's history as a JSON string.
    Groups are found by prompt text, which assumes one process (TRL keeps a prompt's
    num_generations samples together in the same call on a single GPU).
    """
    calls = 0

    def novelty_reward(prompts, completions, history, **kwargs):
        nonlocal calls
        calls += 1
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]

        groups = defaultdict(list)
        for i, p in enumerate(prompts):
            groups[json.dumps(p, sort_keys=True)].append(i)

        rewards = [0.0] * len(texts)
        rows = []
        for idx in groups.values():
            scored = score_group([texts[i] for i in idx], json.loads(history[idx[0]]))
            for i, (tier, r, _) in zip(idx, scored):
                rewards[i] = r
                rows.append({"call": calls, "sample": i, "tier": tier, "reward": r, "completion": texts[i]})

        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        return rewards

    return novelty_reward
