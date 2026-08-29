# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
from __future__ import annotations

import json

import pytest

from train.tool_calling import (
    TOOLS,
    MinedPair,
    ToolCallExample,
    build_corpus,
    build_training_text,
    derive_templated_variants,
    hand_authored_examples,
    mine_factual_pairs,
    parse_tool_call,
    render_tool_call,
    validate_example,
)


def test_there_are_at_least_ninety_hand_authored_seeds():
    """Pins the actual scale, not just "some examples exist" -- the brief was ~100."""
    assert len(hand_authored_examples()) >= 90


def test_every_hand_authored_example_validates():
    """hand_authored_examples() already calls validate_example internally; this test exists so a
    future entry added directly to _HAND_AUTHORED without going through that function is still
    caught."""
    for example in hand_authored_examples():
        validate_example(example)  # must not raise


def test_every_tool_appears_at_least_ten_times_in_the_hand_authored_seeds():
    """A real range across all four tools, not three tools and one token example -- catches the
    hand-authored set collapsing onto a favorite mode."""
    from collections import Counter

    counts = Counter(e.tool for e in hand_authored_examples())
    for tool in TOOLS:
        assert counts[tool] >= 10, f"{tool} only appears {counts[tool]} times"


def test_validate_example_rejects_unknown_tool():
    bad = ToolCallExample(question="q", tool="sarcastic_response", arguments={"answer": "a"})
    with pytest.raises(ValueError, match="unknown tool"):
        validate_example(bad)


def test_validate_example_rejects_missing_argument():
    bad = ToolCallExample(question="q", tool="witty_response", arguments={"answer": "a"})
    with pytest.raises(ValueError, match="missing required argument"):
        validate_example(bad)


def test_validate_example_rejects_extra_argument():
    bad = ToolCallExample(
        question="q", tool="factual_response",
        arguments={"answer": "a", "confidence": "high", "extra": "nope"},
    )
    with pytest.raises(ValueError, match="unexpected argument"):
        validate_example(bad)


def test_validate_example_rejects_bad_enum_value():
    bad = ToolCallExample(
        question="q", tool="factual_response",
        arguments={"answer": "a", "confidence": "medium"},
    )
    with pytest.raises(ValueError, match="not in"):
        validate_example(bad)


def test_validate_example_rejects_empty_question():
    bad = ToolCallExample(
        question="   ", tool="factual_response",
        arguments={"answer": "a", "confidence": "high"},
    )
    with pytest.raises(ValueError, match="question is empty"):
        validate_example(bad)


def test_render_tool_call_produces_the_exact_hermes_parseable_shape():
    """This is the load-bearing format contract: vllm's hermes tool parser scans decoded output
    for a literal `<tool_call>` tag and JSON-parses what's inside it. If this drifts from that
    shape, training teaches the model to emit text the server-side parser won't recognize."""
    text = render_tool_call("witty_response", {"answer": "a joke", "technique": "pun"})
    assert text.startswith("<tool_call>\n")
    assert text.endswith("\n</tool_call>")
    inner = text.removeprefix("<tool_call>\n").removesuffix("\n</tool_call>")
    payload = json.loads(inner)  # must be real, parseable JSON, not just a string that looks like it
    assert payload == {"name": "witty_response", "arguments": {"answer": "a joke", "technique": "pun"}}


def test_render_tool_call_orders_arguments_by_schema_not_call_site():
    """Arguments passed in the 'wrong' order must still serialize in TOOLS' declared order, so
    every training example for one tool has the model see the same argument order every time."""
    text = render_tool_call("witty_response", {"technique": "pun", "answer": "a joke"})
    inner = text.removeprefix("<tool_call>\n").removesuffix("\n</tool_call>")
    assert list(json.loads(inner)["arguments"].keys()) == ["answer", "technique"]


def test_parse_tool_call_is_the_true_inverse_of_render_tool_call():
    for tool, spec in TOOLS.items():
        args = {a: ("high" if a == "confidence" else "pun" if a == "technique" else "value")
                for a in spec["required_args"]}
        text = f"Q: does this round trip?\nAnswer:{render_tool_call(tool, args)}"
        parsed_tool, parsed_args = parse_tool_call(text)
        assert parsed_tool == tool
        assert parsed_args == args


def test_parse_tool_call_returns_none_for_text_with_no_tool_call():
    assert parse_tool_call("Just an ordinary sentence with no tool call in it.") is None


def test_parse_tool_call_returns_none_for_malformed_json_inside_the_tags():
    assert parse_tool_call("<tool_call>\n{not valid json\n</tool_call>") is None


def test_build_training_text_matches_the_dialogue_slices_existing_prompt_convention():
    example = ToolCallExample(
        question="What is the capital of France?", tool="factual_response",
        arguments={"answer": "Paris.", "confidence": "high"},
    )
    text = build_training_text(example)
    assert text.startswith("Q: What is the capital of France?\nAnswer:")
    assert "<tool_call>" in text


# ---------------------------------------------------------------------------------------
# Mining from the real corpus -- the regression tests for the cross-block-matching bug
# ---------------------------------------------------------------------------------------


def test_mine_factual_pairs_needs_the_real_corpus_file(tmp_path):
    missing = tmp_path / "does-not-exist.txt"
    with pytest.raises(FileNotFoundError):
        mine_factual_pairs(dialogue_path=missing)


def test_mining_never_pairs_a_question_with_an_unrelated_later_answer(tmp_path):
    """The bug this test exists to catch: a single whole-file regex with a greedy (or even a
    naively non-greedy but block-crossing) 'context paragraph' group matched a question against
    an unrelated answer many blocks later, whenever the real intervening context didn't fit a
    simple two-paragraph shape. Two blocks here reproduce exactly that shape: the first
    question's own answer sits behind an extra context paragraph, and the file contains a
    SECOND, short, unrelated question/answer pair afterward that a block-crossing match could
    latch onto instead.
    """
    fixture = tmp_path / "dialogue.txt"
    fixture.write_text(
        "Question: What is the capital of France?\n\n"
        "France is a country in Western Europe with a long and storied history.\n\n"
        "Answer: Paris.\n</s>\n\n"
        "Question: What color is grass?\n\n"
        "Answer: Green.\n</s>\n",
        encoding="utf-8",
    )
    pairs = mine_factual_pairs(dialogue_path=fixture, max_pairs=10)
    assert MinedPair(question="What is the capital of France?", answer="Paris.") in pairs
    assert MinedPair(question="What color is grass?", answer="Green.") in pairs
    # The specific failure mode: the France question must NOT have picked up "Green." as its
    # answer just because that Answer: appeared later in the file.
    france = next(p for p in pairs if p.question.startswith("What is the capital"))
    assert france.answer == "Paris."


def test_mining_respects_the_max_pairs_cap():
    pairs = mine_factual_pairs(max_pairs=5)
    assert len(pairs) <= 5


def test_mining_filters_out_long_questions_and_answers():
    fixture_pairs = mine_factual_pairs(max_pairs=200)
    for pair in fixture_pairs:
        assert len(pair.question.split()) <= 20
        assert len(pair.answer.split()) <= 12


def test_mined_pairs_are_real_corpus_content_not_synthetic(tmp_path):
    """A cheap but real provenance check: every mined question must actually occur, verbatim
    modulo whitespace collapsing, in the source file -- if it doesn't, the mining function
    fabricated or corrupted content instead of extracting it."""
    from train.tool_calling import ROOT

    source = (ROOT / "artifacts" / "corpus" / "dialogue.txt").read_text(encoding="utf-8")
    collapsed_source = " ".join(source.split())
    for pair in mine_factual_pairs(max_pairs=20):
        assert pair.question in collapsed_source


def test_derive_templated_variants_covers_all_four_tools_and_all_validate():
    pair = MinedPair(question="What is the capital of France?", answer="Paris.")
    variants = derive_templated_variants(pair)
    assert {v.tool for v in variants} == set(TOOLS)
    for v in variants:
        validate_example(v)  # must not raise
        assert v.provenance == "derived"


def test_derive_templated_variants_keeps_the_real_answer_visible():
    """The derived variants are mechanical, but they must still be traceable to the real fact --
    a witty/absurdist wrapper that drops the actual answer content isn't derived, it's
    hallucinated with extra steps."""
    pair = MinedPair(question="What color is grass?", answer="Green.")
    variants = derive_templated_variants(pair)
    for v in variants:
        rendered_answer = v.arguments.get("answer", "")
        assert "Green." in rendered_answer or "Green" in rendered_answer


def test_build_corpus_puts_hand_authored_examples_first():
    """A consumer that truncates (a smoke test, --limit N) must see the higher-quality
    hand-authored core, not a random mix that happens to include some of it."""
    corpus = build_corpus(max_mined_pairs=10)
    hand_count = len(hand_authored_examples())
    assert all(e.provenance == "hand" for e in corpus[:hand_count])
    assert any(e.provenance == "derived" for e in corpus[hand_count:])


def test_build_corpus_every_example_validates():
    for example in build_corpus(max_mined_pairs=50):
        validate_example(example)  # must not raise
