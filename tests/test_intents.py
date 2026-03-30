"""Tests for chatbot intent classification."""

from chatbot.intents import (
    INTENT_ASSESS,
    INTENT_ASK,
    INTENT_GREETING,
    INTENT_HELP,
    INTENT_RESET,
    classify,
    prefer_knowledge_base_route,
)


def test_faq_risk_factors_goes_to_ask():
    assert (
        classify("What are the risk factors for ED revisits?") == INTENT_ASK
    )


def test_patient_description_goes_to_assess():
    assert (
        classify("72 year old male with COPD, BP 140/90") == INTENT_ASSESS
    )


def test_readmission_risk_question_stays_assess():
    assert (
        classify("What is readmission risk for this patient?") == INTENT_ASSESS
    )


def test_greeting():
    assert classify("Hello!") == INTENT_GREETING


def test_help_command():
    assert classify("help") == INTENT_HELP


def test_reset():
    assert classify("reset") == INTENT_RESET


def test_empty_message_is_help():
    assert classify("   ") == INTENT_HELP


def test_default_unknown_is_ask():
    assert classify("mumble") == INTENT_ASK


def test_prefer_knowledge_base_route_faq():
    assert prefer_knowledge_base_route("What are the risk factors for ED revisits?")


def test_prefer_knowledge_base_route_false_for_readmission_cue():
    assert not prefer_knowledge_base_route("What is readmission risk?")


def test_prefer_knowledge_base_route_false_with_age():
    assert not prefer_knowledge_base_route("What is COPD in a 65 year old?")
