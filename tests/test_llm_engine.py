"""Tests for optional LLM synthesis in ChatEngine (mocked HTTP)."""

from __future__ import annotations

import pytest

from chatbot.engine import ChatEngine


@pytest.fixture
def engine_skip_model_load(monkeypatch: pytest.MonkeyPatch) -> ChatEngine:
    monkeypatch.setattr(ChatEngine, "_load_models", lambda self: None)
    return ChatEngine()


def test_ask_template_when_llm_disabled(
    engine_skip_model_load: ChatEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chatbot.engine.rag_available", lambda _d: True)
    monkeypatch.setattr(
        "chatbot.engine.retrieve",
        lambda _art, _q, top_k=3: [
            {"score": 0.9, "source": "doc.md", "excerpt": "## Topic\n\nBody text."},
        ],
    )
    monkeypatch.setattr("chatbot.engine.llm_configured", lambda: False)

    out = engine_skip_model_load._ask("What is discharge planning?")
    assert "### Knowledge Base Results" in out
    assert "Body text." in out


def test_ask_uses_llm_when_enabled(
    engine_skip_model_load: ChatEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chatbot.engine.rag_available", lambda _d: True)
    monkeypatch.setattr(
        "chatbot.engine.retrieve",
        lambda _art, _q, top_k=3: [
            {"score": 0.9, "source": "kb.md", "excerpt": "Evidence line."},
        ],
    )
    monkeypatch.setattr("chatbot.engine.llm_configured", lambda: True)
    monkeypatch.setattr(
        "chatbot.engine.llm_complete",
        lambda _s, _u: "**Synthesized** answer from passages.",
    )

    out = engine_skip_model_load._ask("What is COPD?")
    assert "**Synthesized** answer" in out
    assert "demonstration only" in out.lower()


def test_ask_falls_back_when_llm_returns_none(
    engine_skip_model_load: ChatEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chatbot.engine.rag_available", lambda _d: True)
    monkeypatch.setattr(
        "chatbot.engine.retrieve",
        lambda _art, _q, top_k=3: [
            {"score": 0.8, "source": "x.md", "excerpt": "Fallback chunk."},
        ],
    )
    monkeypatch.setattr("chatbot.engine.llm_configured", lambda: True)
    monkeypatch.setattr("chatbot.engine.llm_complete", lambda _s, _u: None)

    out = engine_skip_model_load._ask("Why revisits?")
    assert "### Knowledge Base Results" in out
    assert "Fallback chunk." in out


@pytest.fixture
def engine_assess(monkeypatch: pytest.MonkeyPatch) -> ChatEngine:
    def _fake_load(self: ChatEngine) -> None:
        self.models["readmission"] = {}

    monkeypatch.setattr(ChatEngine, "_load_models", _fake_load)
    monkeypatch.setattr(
        ChatEngine,
        "_run_predictions",
        lambda self: {"readmission": 0.22},
    )
    return ChatEngine()


def test_assess_template_when_llm_disabled(
    engine_assess: ChatEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chatbot.engine.extract_all", lambda _m: {"AGE": 66.0, "CHF": 1.0})
    monkeypatch.setattr("chatbot.engine.rag_available", lambda _d: False)
    monkeypatch.setattr("chatbot.engine.llm_configured", lambda: False)

    out = engine_assess._assess("66 year old with CHF")
    assert "### Patient Summary" in out
    assert "### Readmission Risk" in out
    assert "22.0%" in out or "22%" in out
    assert "new patient" in out.lower()


def test_assess_uses_llm_when_enabled(
    engine_assess: ChatEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chatbot.engine.extract_all", lambda _m: {"AGE": 50.0})
    monkeypatch.setattr("chatbot.engine.rag_available", lambda _d: False)
    monkeypatch.setattr("chatbot.engine.llm_configured", lambda: True)
    monkeypatch.setattr(
        "chatbot.engine.llm_complete",
        lambda _s, _u: "### Summary\n\nLLM narrative with **22.0%** risk.",
    )

    out = engine_assess._assess("50 year old male")
    assert "LLM narrative" in out
    assert "demonstration only" in out.lower()


def test_assess_falls_back_when_llm_returns_none(
    engine_assess: ChatEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chatbot.engine.extract_all", lambda _m: {"AGE": 40.0})
    monkeypatch.setattr("chatbot.engine.rag_available", lambda _d: False)
    monkeypatch.setattr("chatbot.engine.llm_configured", lambda: True)
    monkeypatch.setattr("chatbot.engine.llm_complete", lambda _s, _u: None)

    out = engine_assess._assess("40yo")
    assert "### Patient Summary" in out
    assert "### Readmission Risk" in out
