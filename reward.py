#!/usr/bin/env python3
"""
Novelty reward for GRPO with one command per response.

The model answers with one JSON object: {"command": str, "parameters": [str, ...]}.
The `command` field is the command's structure and the parameters are its values,
so no canonicalization rules are needed. Each completion is scored against the
history shown in its own prompt:

    not one JSON object / empty command      broken          -6
    same command and same parameters seen    repeat          -1
    command seen, parameters new             new_value       -1
    command never seen                       new_structure   +5 / k

k is the number of completions in the same group (same prompt) that produced
the same new command, so a group that all finds the same thing shares one bonus.
Only new command names are rewarded: new parameters on a known command score like a
repeat (anything above -1 would still be reinforced relative to repeats in a GRPO group).
new_value stays a separate tier so the logs show how often the model tries it.
Novelty of parameters is a separate problem, deliberately left out.
Gibberish commands count like any other: only novelty is rewarded.
`plausible_syntax` is a rough check for that, used only in logging, never in the reward.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

REWARDS = {"broken": -6.0, "repeat": -1.0, "new_value": -1.0, "new_structure": 5.0}
TIERS = list(REWARDS)

# Korad-style syntax: optional '*', upper-case letters, optional channel digits, then '?' or ':<value>'
# (VSET1:, ISET1?, OUT1, *IDN?). A proxy for "not gibberish" on this target, not a correctness check.
_PLAUSIBLE = re.compile(r"^\*?[A-Z]{2,}\d*(\?|:.*)?$")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")

_WORDS = re.compile(r"[A-Z]+")
# The documented KA3005P protocol; command and value may be split between `command` and `parameters`.
_KA3005P = re.compile(r"^(VSET1|ISET1)(:[\d.]+|\?)?$|^(VOUT1|IOUT1|STATUS|\*IDN)\??$"
                      r"|^(OUT|OVP|OCP|BEEP)[01]?$|^(SAV|RCL)[1-5]?$")


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


def plausible_syntax(command: str) -> bool:
    return bool(_PLAUSIBLE.match(command))


def head_words(command: str, n: int = 3) -> str:
    """The first n words of a command name (letter runs, upper-cased), e.g. 'SET OUTPUT VOLTAGE'.

    Counting distinct heads instead of distinct names discounts padding: names made new
    only by appending words ('SET OUTPUT VOLTAGE RAMP ... WITH SLEW LIMIT') share a head.
    """
    return " ".join(_WORDS.findall(command.upper())[:n])


def ka3005p_command(command: str) -> bool:
    """True if the name is in the documented KA3005P command set (VSET1, ISET1?, OUT1, *IDN?, ...)."""
    return bool(_KA3005P.match(command.strip().upper().replace(" ", "")))


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


def make_reward_fn(log_path: Optional[Path] = None, sample_every: int = 0, sample_groups: int = 2,
                   history_tail: int = 5):
    """Build the TRL reward function. Every scored completion is appended to `log_path` (JSONL) if given.

    The dataset must have a `history` column holding the prompt's history as a JSON string.
    Groups are found by prompt text, which assumes one process (TRL keeps a prompt's
    num_generations samples together in the same call on a single GPU).

    When TRL passes `log_extra` / `log_metric` (TRL >= 1.x), each completion's tier and parsed
    command are added to TRL's completions table, and these are logged next to `reward`:

        tier/<tier>_rate                         share of completions in each tier
        novelty/distinct_new_commands            different new commands in the whole batch
        novelty/distinct_new_commands_per_group  the same, averaged over prompts
        novelty/nonstandard_new_command_rate     share of new commands failing plausible_syntax

    With `sample_every` > 0 and an active wandb run, `sample_groups` whole groups are added to a
    growing `samples` W&B table at step 1 and every `sample_every` steps after.
    """
    calls = 0
    sample_rows = []
    last_sampled = None

    def novelty_reward(prompts, completions, history, trainer_state=None, log_extra=None, log_metric=None,
                       **kwargs):
        nonlocal calls, last_sampled
        calls += 1
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]

        groups = defaultdict(list)
        for i, p in enumerate(prompts):
            groups[json.dumps(p, sort_keys=True)].append(i)

        rewards = [0.0] * len(texts)
        tiers = [None] * len(texts)
        parsed = [None] * len(texts)
        rows = []
        for idx in groups.values():
            scored = score_group([texts[i] for i in idx], json.loads(history[idx[0]]))
            for i, (tier, r, p) in zip(idx, scored):
                rewards[i], tiers[i], parsed[i] = r, tier, p
                rows.append({"call": calls, "sample": i, "tier": tier, "reward": r, "completion": texts[i]})

        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

        if log_extra is not None:
            log_extra("tier", tiers)
            log_extra("command", [p[0] if p else None for p in parsed])
            log_extra("parameters", [json.dumps(p[1]) if p else None for p in parsed])

        if log_metric is not None:
            for t in TIERS:
                log_metric(f"tier/{t}_rate", tiers.count(t) / len(tiers))
            new = [(i, parsed[i][0]) for i in range(len(texts)) if tiers[i] == "new_structure"]
            log_metric("novelty/distinct_new_commands", len({c for _, c in new}))
            log_metric("novelty/distinct_new_commands_per_group",
                       sum(len({parsed[i][0] for i in idx if tiers[i] == "new_structure"}) for idx in groups.values())
                       / len(groups))
            if new:
                log_metric("novelty/nonstandard_new_command_rate",
                           sum(not plausible_syntax(c) for _, c in new) / len(new))

        # These completions are trained on in the coming update, which TRL logs as global_step + 1.
        step = trainer_state.global_step + 1 if trainer_state is not None else calls
        if sample_every and step != last_sampled and (step == 1 or step % sample_every == 0):
            last_sampled = step
            _log_samples(step, list(groups.values())[:sample_groups], texts, history, tiers, parsed, rewards)

        return rewards

    def _log_samples(step, sampled_groups, texts, history, tiers, parsed, rewards):
        import wandb
        if wandb.run is None:
            return
        for g, idx in enumerate(sampled_groups):
            h = json.loads(history[idx[0]])
            tail = "\n".join(json.dumps(x) for x in h[-history_tail:])
            for i in idx:
                sample_rows.append([step, g, len(h), tail, texts[i], tiers[i],
                                    parsed[i][0] if parsed[i] else None,
                                    json.dumps(parsed[i][1]) if parsed[i] else None, rewards[i]])
        columns = ["step", "group", "history_len", "history_tail", "completion", "tier", "command", "parameters",
                   "reward"]
        # Re-logged whole each time so one table holds every snapshot; commit=False attaches it to the next train log.
        wandb.log({"samples": wandb.Table(columns=columns, data=sample_rows)}, commit=False)

    return novelty_reward
