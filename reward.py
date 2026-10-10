#!/usr/bin/env python3
"""
Novelty reward for GRPO with one command per response.

The model answers with one JSON object: {"command": str, "parameters": [str, ...]}.
Only the `command` field is judged; parameters are ignored. The command is turned into
a word sequence (command_words) and compared with every command in the history shown
in its own prompt by ROUGE-L (rouge_l: 2 x longest common subsequence of words / total words):

    not one JSON object / empty command            broken          -10
    same words as a history command (ROUGE-L 1)    repeat           -6   (also with new parameters)
    ROUGE-L >= threshold (default 0.7) to one      near_duplicate   -3
    ROUGE-L below the threshold to all of them     new_structure    +5 / k

k is the number of new completions in the same group (same prompt) whose words are within
the threshold of this one (itself included), so a group that all finds the same thing shares
one bonus. Near-duplicates catch the ways v1-v3 models made names "new": padding
(SET OUTPUT VOLTAGE -> ... RAMP RATE WITH SLEW LIMIT), copying the tail after a swapped
word, CamelCase or glued spellings of a known name, and counters (C1111, C1112).
Novelty of parameters is a separate problem, deliberately left out.
Gibberish commands count like any other: only novelty is rewarded.
`plausible_syntax` is a rough check for that, used only in logging, never in the reward.
"""

import gzip
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REWARDS = {"broken": -10.0, "repeat": -6.0, "near_duplicate": -3.0, "new_structure": 5.0}
TIERS = list(REWARDS)
SIM_THRESHOLD = 0.7                            # ROUGE-L at or above this is a near-duplicate (reward and eval)
EVAL_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)    # distinct-command counts logged in eval; SIM_THRESHOLD is the main one

# command_words: names are cut into words, then compared word by word.
_SEPARATORS = re.compile(r"[^A-Za-z0-9]+")
# CamelCase and acronym boundaries (SetVoltage, HTTPServer) and letter/digit boundaries (VSET1).
_PIECES = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
# Not AT: it is the prefix of AT modem commands (AT+CIPSTART).
STOPWORDS = frozenset({"A", "AN", "THE", "OF", "TO", "AND", "OR", "FOR", "WITH", "IN", "ON", "BY", "FROM"})
MAX_WORDS = 6               # words compared per name; padding beyond this cannot make a name new
GLUED_MIN_LETTERS = 8       # shorter runs (VSET, OVPSET) are left whole
DICTIONARY = Path(__file__).with_name("command_words.txt.gz")   # built once by build_dictionary.py

# Korad-style syntax: optional '*', upper-case letters, optional channel digits, then '?' or ':<value>'
# (VSET1:, ISET1?, OUT1, *IDN?). A proxy for "not gibberish" on this target, not a correctness check.
_PLAUSIBLE = re.compile(r"^\*?[A-Z]{2,}\d*(\?|:.*)?$")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")

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


def ka3005p_command(command: str) -> bool:
    """True if the name is in the documented KA3005P command set (VSET1, ISET1?, OUT1, *IDN?, ...)."""
    return bool(_KA3005P.match(command.strip().upper().replace(" ", "")))


def explicit_pieces(name: str) -> List[str]:
    """Cut a name where its own spelling marks word breaks: symbols, spaces, case and digit changes.

    'SOUR:VOLT' -> SOUR VOLT, 'SetOutputVoltage' -> Set Output Voltage, 'VSET1' -> VSET 1.
    """
    return [w for part in _SEPARATORS.split(name) for w in _PIECES.findall(part)]


@lru_cache(maxsize=1)
def _dictionary():
    """Word costs for _segment (cost = log(rank x log(size)), as in wordninja) and the models' own words."""
    if not DICTIONARY.exists():
        raise FileNotFoundError(f"{DICTIONARY} is missing; build it with build_dictionary.py")
    with gzip.open(DICTIONARY, "rt", encoding="utf-8") as f:
        header, *words = f.read().split("\n")
    model_words = frozenset(words[:int(header.split()[1])])   # header: '#model_words <n>'
    costs = {}
    for i, w in enumerate(words):
        costs.setdefault(w, math.log((i + 1) * math.log(len(words))))
    return costs, max(map(len, words)), model_words


def _segment(word: str) -> List[str]:
    """Split a glued word ('GETBATTSTATMAX') into dictionary words by minimum total cost.

    The split is used only if every piece has 3+ letters or is a 2-letter word the models
    themselves wrote separately; otherwise the word stays whole.
    """
    costs, longest, model_words = _dictionary()
    s = word.lower()
    best = [(0.0, 0)]   # best[i] = (cost of s[:i], length of its last piece)
    for i in range(1, len(s) + 1):
        best.append(min((best[i - k][0] + costs.get(s[i - k:i], math.inf), k) for k in range(1, min(i, longest) + 1)))
    pieces, i = [], len(s)
    while i > 0:
        k = best[i][1]
        pieces.append(s[i - k:i])
        i -= k
    pieces.reverse()
    if len(pieces) > 1 and all(len(p) >= 3 or (len(p) == 2 and p in model_words) for p in pieces):
        return [p.upper() for p in pieces]
    return [word]


@lru_cache(maxsize=500_000)
def command_words(name: str) -> Tuple[str, ...]:
    """The words a command name is compared by.

    Cut at symbols, case changes and digits (explicit_pieces); digit runs become '#', so values
    and counters in a name (VSET1, C1112) do not make it new; glued words of 8+ letters are split
    with the dictionary (GETBATTSTATMAX -> GET BATT STAT MAX); words are upper-cased, a plural S
    is dropped, repeated words and STOPWORDS are removed, and the first MAX_WORDS are kept.
    'Set the Output Voltages' and 'set_output_voltage' both give ('SET', 'OUTPUT', 'VOLTAGE').
    """
    words = []
    for piece in explicit_pieces(name):
        if piece.isdigit():
            words.append("#")
            continue
        piece = piece.upper()
        words += _segment(piece) if len(piece) >= GLUED_MIN_LETTERS else [piece]
    out = []
    for w in words:
        if len(w) > 3 and w.endswith("S") and not w.endswith(("SS", "US", "IS")):
            w = w[:-1]
        if w not in STOPWORDS and w not in out:
            out.append(w)
    return tuple((out or words[:1] or [name.strip().upper()])[:MAX_WORDS])


def rouge_l(a: Sequence[str], b: Sequence[str]) -> float:
    """ROUGE-L F1 of two word sequences: 2 x longest common subsequence / (len a + len b); 1 means identical."""
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return 2 * prev[-1] / (len(a) + len(b))


def max_similarity(words: Sequence[str], others) -> float:
    """Highest rouge_l between `words` and any of `others` (0 if there are none)."""
    return max((rouge_l(words, o) for o in others), default=0.0)


class DistinctCommands:
    """Commands that count as different at `threshold`, kept greedily in list order.

    A command is kept if its ROUGE-L to every command kept before it is below the threshold;
    len() is the number of distinct commands. Used for the eval counts at EVAL_THRESHOLDS.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.kept = []

    def add(self, command: str) -> bool:
        w = command_words(command)
        if max_similarity(w, self.kept) < self.threshold:
            self.kept.append(w)
            return True
        return False

    def __len__(self):
        return len(self.kept)


def score_group(texts: List[str], history: List[dict],
                threshold: float = SIM_THRESHOLD) -> List[Tuple[str, float, Optional[tuple]]]:
    """Score the completions of one prompt. Returns (tier, reward, parsed) per completion, in order."""
    seen = [command_words(h["command"]) for h in history]
    seen_set = set(seen)
    parsed = [parse_completion(t) for t in texts]
    words = [command_words(p[0]) if p else None for p in parsed]

    tiers = []
    for p, w in zip(parsed, words):
        if p is None:
            tiers.append("broken")
        elif w in seen_set:
            tiers.append("repeat")
        elif max_similarity(w, seen) >= threshold:
            tiers.append("near_duplicate")
        else:
            tiers.append("new_structure")

    new = [w for w, t in zip(words, tiers) if t == "new_structure"]
    out = []
    for p, w, t in zip(parsed, words, tiers):
        if t == "new_structure":
            k = sum(rouge_l(w, o) >= threshold for o in new)   # includes itself
            out.append((t, REWARDS[t] / k, p))
        else:
            out.append((t, REWARDS[t], p))
    return out


def make_reward_fn(log_path: Optional[Path] = None, sample_every: int = 0, sample_groups: int = 2,
                   history_tail: int = 5, threshold: float = SIM_THRESHOLD):
    """Build the TRL reward function. Every scored completion is appended to `log_path` (JSONL) if given.

    The dataset must have a `history` column holding the prompt's history as a JSON string.
    Groups are found by prompt text, which assumes one process (TRL keeps a prompt's
    num_generations samples together in the same call on a single GPU).

    When TRL passes `log_extra` / `log_metric` (TRL >= 1.x), each completion's tier and parsed
    command are added to TRL's completions table, and these are logged next to `reward`:

        tier/<tier>_rate                         share of completions in each tier
        novelty/distinct_new_commands            different new commands (distinct command_words) in the whole batch
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
            scored = score_group([texts[i] for i in idx], json.loads(history[idx[0]]), threshold)
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
            log_metric("novelty/distinct_new_commands", len({command_words(c) for _, c in new}))
            log_metric("novelty/distinct_new_commands_per_group",
                       sum(len({command_words(parsed[i][0]) for i in idx if tiers[i] == "new_structure"})
                           for idx in groups.values())
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
