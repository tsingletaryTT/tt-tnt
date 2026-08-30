# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
from __future__ import annotations

from scripts.eval_tool_calling import PROBES, classify_completion
from train.tool_calling import render_tool_call

Q = "What is the capital of France?"


def test_a_well_formed_registered_call_passes_all_three_gates():
    text = " " + render_tool_call("factual_response", {"answer": "Paris.", "confidence": "high"})
    v = classify_completion(text, Q)
    assert v == {"emitted": True, "parsed": True, "schema_valid": True,
                 "tool": "factual_response"}


def test_ordinary_prose_fails_every_gate():
    v = classify_completion("The capital of France is Paris.", Q)
    assert v["emitted"] is False and v["parsed"] is False and v["schema_valid"] is False
    assert v["tool"] is None


def test_a_call_naming_an_UNREGISTERED_tool_parses_but_is_not_schema_valid():
    """The gate that matters most: a model that invents `sarcastic_response` produces JSON a
    lenient consumer would accept. It must count as parsed (it IS a tool call) and NOT as
    schema-valid (it is not one of ours)."""
    text = '<tool_call>\n{"name": "sarcastic_response", "arguments": {"answer": "Oh, Paris."}}\n</tool_call>'
    v = classify_completion(text, Q)
    assert v["parsed"] is True
    assert v["schema_valid"] is False
    assert v["tool"] == "sarcastic_response"


def test_a_registered_tool_MISSING_a_required_argument_is_not_schema_valid():
    text = '<tool_call>\n{"name": "witty_response", "arguments": {"answer": "F."}}\n</tool_call>'
    v = classify_completion(text, Q)
    assert v["parsed"] is True and v["schema_valid"] is False


def test_a_registered_tool_with_an_ILLEGAL_ENUM_value_is_not_schema_valid():
    text = ('<tool_call>\n{"name": "factual_response", "arguments": '
            '{"answer": "Paris.", "confidence": "medium"}}\n</tool_call>')
    v = classify_completion(text, Q)
    assert v["parsed"] is True and v["schema_valid"] is False


def test_a_truncated_tool_call_counts_as_emitted_but_not_parsed():
    """The most likely real failure at 120 max_new_tokens: the model starts a call and runs
    out of budget. That must show up as emitted-but-unparseable, which is a different
    diagnosis from 'never tried'."""
    text = '<tool_call>\n{"name": "absurdist_response", "arguments": {"answer": "The capital'
    v = classify_completion(text, Q)
    assert v["emitted"] is True
    assert v["parsed"] is False
    assert v["schema_valid"] is False


def test_malformed_json_inside_the_tags_counts_as_emitted_but_not_parsed():
    v = classify_completion("<tool_call>\n{not json at all}\n</tool_call>", Q)
    assert v["emitted"] is True and v["parsed"] is False


def test_every_registered_tool_round_trips_to_schema_valid():
    """No registered tool may be un-scoreable by its own evaluator."""
    from train.tool_calling import TOOLS

    for tool, spec in TOOLS.items():
        args = {a: ("high" if a == "confidence" else "pun" if a == "technique" else "v")
                for a in spec["required_args"]}
        v = classify_completion(render_tool_call(tool, args), Q)
        assert v["schema_valid"] is True, f"{tool} did not validate against its own schema"


def test_probes_contain_both_seen_and_unseen_questions():
    """A battery of only-memorised questions cannot distinguish learning the FORMAT from
    reproducing training rows."""
    provenances = {p for _, p in PROBES}
    assert provenances == {"seen", "unseen"}
    assert sum(1 for _, p in PROBES if p == "unseen") >= 3


def test_probe_provenance_labels_are_accurate_against_the_actual_seed_set():
    """The 'seen'/'unseen' labels are load-bearing -- a mislabelled probe would silently turn
    a generalisation claim into a memorisation claim. Checked against the real hand-authored
    set rather than trusted."""
    from train.tool_calling import hand_authored_examples

    seed_questions = {e.question for e in hand_authored_examples()}
    for question, provenance in PROBES:
        if provenance == "seen":
            assert question in seed_questions, f"{question!r} labelled seen but is not a seed"
        else:
            assert question not in seed_questions, (
                f"{question!r} labelled unseen but IS in the hand-authored seeds"
            )
