"""Keyword-based intent classifier for the ED Risk chatbot."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

INTENT_GREETING = "greeting"
INTENT_ASSESS = "assess"
INTENT_UPDATE = "update"
INTENT_ASK = "ask"
INTENT_HELP = "help"
INTENT_RESET = "reset"


@dataclass
class _IntentRule:
    name: str
    patterns: List[re.Pattern]
    priority: int = 0


_RULES: List[_IntentRule] = [
    _IntentRule(
        INTENT_RESET,
        [re.compile(p, re.I) for p in [
            r"\b(new patient|reset|clear|start over)\b",
        ]],
        priority=90,
    ),
    _IntentRule(
        INTENT_HELP,
        [re.compile(p, re.I) for p in [
            r"^help$",
            r"\bwhat can you do\b",
            r"\bcommands\b",
            r"\bhow do (i|I) use\b",
        ]],
        priority=80,
    ),
    _IntentRule(
        INTENT_GREETING,
        [re.compile(p, re.I) for p in [
            r"^(hi|hello|hey|greetings|good (morning|afternoon|evening))[\s!.]*$",
        ]],
        priority=70,
    ),
    _IntentRule(
        INTENT_ASK,
        [re.compile(p, re.I) for p in [
            r"^(what|why|how|when|who|tell me|explain|describe)\b",
            r"\bwhat is\b",
            r"\btell me about\b",
        ]],
        priority=30,
    ),
    _IntentRule(
        INTENT_UPDATE,
        [re.compile(p, re.I) for p in [
            r"\b(actually|change|update|correct|set)\b.*(to|is|=)\b",
        ]],
        priority=50,
    ),
    _IntentRule(
        INTENT_ASSESS,
        [re.compile(p, re.I) for p in [
            r"\d+\s*(yr|year|yo|y/?o)\b",
            r"\b(patient|pt)\b",
            r"\b(male|female)\b",
            r"\b(age|temp|pulse|bp|pain|lov|chronic|arriv|triage)\b",
            r"\b(assess|predict|evaluate|risk|score)\b",
        ]],
        priority=40,
    ),
]


def classify(message: str) -> str:
    """Return the best-matching intent name for *message*."""
    msg = message.strip()
    if not msg:
        return INTENT_HELP

    hits: List[Tuple[int, str]] = []
    for rule in _RULES:
        for pat in rule.patterns:
            if pat.search(msg):
                hits.append((rule.priority, rule.name))
                break

    if not hits:
        return INTENT_ASK

    hits.sort(key=lambda t: -t[0])
    return hits[0][1]
