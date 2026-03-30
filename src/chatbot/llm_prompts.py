"""System and user prompt templates for grounded LLM replies."""

from __future__ import annotations

import json
from typing import Any, Dict, List

DISCLAIMER = (
    "*This assistant is for demonstration only and is not a substitute for "
    "clinical judgment or professional medical advice.*"
)


def ask_system_prompt() -> str:
    return (
        "You are a clinical documentation assistant for an ED readmission-risk demo. "
        "Answer ONLY using the knowledge-base passages provided in the user message. "
        "If the passages do not contain enough information, say so briefly. "
        "When you paraphrase, tie claims to the source filename shown for each passage. "
        "Use clear Markdown (headings, bullets). "
        "Do not invent citations or clinical facts not supported by the passages."
    )


def ask_user_prompt(user_question: str, passages: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        src = p.get("source", "unknown")
        ex = (p.get("excerpt") or "").strip()
        blocks.append(f"### Passage {i} (source: {src})\n\n{ex}")
    joined = "\n\n".join(blocks)
    return (
        f"## User question\n\n{user_question.strip()}\n\n"
        f"## Knowledge base passages\n\n{joined}\n\n"
        "Answer the question using only the passages above."
    )


def assess_system_prompt() -> str:
    return (
        "You are a clinical documentation assistant for an ED readmission-risk demo. "
        "You will receive structured patient data, an authoritative readmission risk "
        "estimate (probability and band), optional NHAMCS condition statistics, and "
        "knowledge-base excerpts for recommendations.\n\n"
        "Requirements:\n"
        "- Begin by restating the readmission risk EXACTLY as given (same percentage "
        "and High/Moderate/Low label).\n"
        "- Summarize the patient using only fields provided in the JSON; do not add "
        "vitals, conditions, or history that are not listed.\n"
        "- For recommendations, ground every bullet in the knowledge-base excerpts; "
        "if an excerpt does not support a point, omit it.\n"
        "- Use concise Markdown.\n"
        "- End with a short line that this output is for demonstration, not clinical advice."
    )


def assess_user_prompt(ctx: Dict[str, Any]) -> str:
    body = json.dumps(ctx, indent=2, default=str)
    return (
        "## Context (authoritative — do not change numbers)\n\n"
        f"```json\n{body}\n```\n\n"
        "Produce a clinician-facing summary: risk line first, then patient summary, "
        "then bullet recommendations tied to the excerpts."
    )
