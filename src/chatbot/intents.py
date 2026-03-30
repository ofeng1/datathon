"""Keyword-based intent classifier for the ED Risk chatbot."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set, Tuple

INTENT_GREETING = "greeting"
INTENT_ASSESS = "assess"
INTENT_UPDATE = "update"
INTENT_ASK = "ask"
INTENT_HELP = "help"
INTENT_RESET = "reset"

# Prediction / readmission phrasing (not e.g. "risk factors" FAQ).
_READMISSION_RISK_CUE = re.compile(
    r"\breadmission\s+risk\b|"
    r"\brisk\s+of\s+readmission\b|"
    r"\brisk\s+score\b|"
    r"\brisk\s+assessment\b",
    re.I,
)

_QUESTION_LEAD = re.compile(
    r"^(?:what|why|how|when|who|tell me|explain|describe)\b",
    re.I,
)


def _strong_patient_cues(msg: str) -> bool:
    if re.search(r"\d+\s*(?:yr|year|yo|y/?o)\b", msg, re.I):
        return True
    if re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\b", msg):
        return True
    if re.search(
        r"\b(?:temp|temperature)\b.*\d|\d.*\b(?:temp|temperature)\b", msg, re.I
    ):
        return True
    if re.search(r"\bpulse\b.*\d|\d.*\bpulse\b", msg, re.I):
        return True
    if re.search(r"\bpain\b.*\d\s*/\s*\d|\b\d\s*/\s*10\b", msg, re.I):
        return True
    return False


def prefer_knowledge_base_route(message: str) -> bool:
    """True when the message should use KB/RAG rather than patient extraction if ambiguous."""
    msg = message.strip()
    if not msg:
        return False
    if _READMISSION_RISK_CUE.search(msg):
        return False
    if not _QUESTION_LEAD.search(msg):
        return False
    if _strong_patient_cues(msg):
        return False
    return True


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
            r"\b(assess|predict|evaluate)\b",
            r"\breadmission\s+risk\b",
            r"\brisk\s+of\s+readmission\b",
            r"\brisk\s+score\b",
            r"\brisk\s+assessment\b",
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
    matched_names: Set[str] = set()
    for rule in _RULES:
        for pat in rule.patterns:
            if pat.search(msg):
                hits.append((rule.priority, rule.name))
                matched_names.add(rule.name)
                break

    if not hits:
        return INTENT_ASK

    hits.sort(key=lambda t: -t[0])
    winner = hits[0][1]

    if (
        winner == INTENT_ASSESS
        and INTENT_ASK in matched_names
        and prefer_knowledge_base_route(msg)
    ):
        return INTENT_ASK

    return winner
