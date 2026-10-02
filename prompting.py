#!/usr/bin/env python3
"""Render the one-command prompt. The history format must stay stable so prompts remain comparable."""

import json
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_ENV = Environment(loader=FileSystemLoader(Path(__file__).resolve().parent), undefined=StrictUndefined)


def format_history(history: List[dict]) -> str:
    """One JSON object per line, in the order the commands were sent."""
    return "\n".join(json.dumps({"command": h["command"], "parameters": h["parameters"]}) for h in history)


def build_messages(target: str, interface: str, history: List[dict], template: str = "template.jinja") -> list:
    text = _ENV.get_template(template).render(TARGET=target, INTERFACE=interface, HISTORY=format_history(history))
    return [{"role": "user", "content": text}]
