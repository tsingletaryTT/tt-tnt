# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Wiring tests for scripts/derive_skits.py's build_skit_example.

STORY is deliberately NOT the dialogue-with-attribution shape ('"..." said X.'): that
shape is the documented split_sentences over-split (train/skit.py:31-38) and derive_skit
genuinely returns None on it (verified by hand against the merged train/skit.py before
writing this fixture) -- turn 1 becomes just the quoted line and turn 2 becomes just
"said her friend.", which share no content word. That is real corpus drop-rate pressure,
correctly reported in the derivation stats below, but it must not also break this fixture:
a fixture that can't even produce a Skit tests nothing. Every turn here instead carries a
plain declarative sentence with a word bridging it to the next, so accept/add succeed at
every one of the three model turns.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.derive_skits import TILE, build_skit_example
from train.skit import MODEL_TURNS, PARTNER_TURNS, derive_skit, skit_segments

STORY = ("Lily found a shiny rock. She showed it to her friend. "
         "The rock sparkled on the windowsill. "
         "Her friend loved the windowsill shine. "
         "The shine made a rainbow appear. "
         "They admired the rainbow glow. "
         "The glow lasted through the evening.")

_IDS = {}


class _Tok:
    """Faithful, deterministic, and honours add_special_tokens.

    Deterministic on purpose: builtins hash() is randomised per process, and a mock that
    ignores add_special_tokens let a spurious-BOS bug pass every stage-1 test.
    """
    pad_token_id = 0
    BOS = 1

    def encode(self, s, add_special_tokens=True):
        ids = [_IDS.setdefault(w, len(_IDS) + 2) for w in s.split()]
        return ([self.BOS] + ids) if add_special_tokens else ids


def _skit():
    s = derive_skit(STORY, story_id=0, idf={"windowsill": 1.0, "rainbow": 1.0},
                    intensity=lambda t: 0.0)
    assert s is not None
    return s


def _segment_spans(skit, tok, with_think):
    """Rebuild (start, end_exclusive, supervised) token spans, INDEPENDENTLY of
    build_skit_example's own internal state -- this walks the same public
    `skit_segments` contract build_skit_example consumes, but keeps its own running
    position counter and its own list of spans, so it is not just re-reading a value
    build_skit_example already computed. That independence is the point: a test that
    asked build_skit_example for its own `supervised` list back would validate that the
    function agrees with itself, not that its labels match the segment boundaries.
    """
    spans = []
    pos = 0
    first = True
    for text, sup in skit_segments(skit):
        if not with_think and text.lstrip().startswith("<think>"):
            continue
        seg = tok.encode(text, add_special_tokens=first)
        first = False
        spans.append((pos, pos + len(seg), sup))
        pos += len(seg)
    return spans


def test_labels_are_pre_shifted_at_every_supervised_position():
    """ttml compares logits[t] to labels[t] with no internal shift. The HF convention
    silently trained two arms against wrong targets in stage 1.

    SURVIVING MUTANT this test used to miss: `supervised[t]` instead of
    `supervised[t + 1]` in the label rule keeps every VALUE equal to `ids[t + 1]` (the
    old loop below still passes) while shifting WHICH positions are supervised by one.
    That silently drops all three unsupervised->supervised transitions (the exact thing
    this module's docstring calls "the transition being trained") and leaks one leading
    token of each partner/prefix span into the loss as a supervised target. A test that
    only checks the value at positions ALREADY KNOWN to be non-(-100) can never see a
    bug in which positions those are -- it has to check the SET.
    """
    skit, tok = _skit(), _Tok()
    ex = build_skit_example(skit, tok, with_think=True, pad_token_id=0)
    ids, labs = ex["input_ids"], ex["labels"]
    assert len(ids) == len(labs)

    # Old (value-only) check, kept: every non-masked label is still the next id.
    for t, v in enumerate(labs):
        if v != -100:
            assert t + 1 < len(ids), "a supervised position must have a next token"
            assert v == ids[t + 1], f"position {t} is not pre-shifted"

    # New (set) check: supervised positions are EXACTLY where the spec says they must
    # be -- t is supervised iff t+1 falls inside a supervised segment. Recomputed here
    # independently from the segment spans, not from build_skit_example's own labels.
    spans = _segment_spans(skit, tok, with_think=True)
    total_unpadded = spans[-1][1] if spans else 0
    supervised_positions = {i for start, end, sup in spans if sup for i in range(start, end)}
    expected = {t for t in range(total_unpadded - 1) if (t + 1) in supervised_positions}
    actual = {t for t, v in enumerate(labs) if v != -100}
    assert actual == expected, (
        f"supervised positions {sorted(actual)} != expected {sorted(expected)} -- "
        f"the mutant shifts WHICH positions are labelled without changing their values")

    # Boundary form of the same check, spelled out per the brief: at every
    # unsupervised->supervised transition, the LAST unsupervised position's label must
    # be the FIRST token of the following supervised segment.
    for i in range(len(spans) - 1):
        _, b_end, b_sup = spans[i]
        a2_start, _, a2_sup = spans[i + 1]
        if not b_sup and a2_sup:
            assert labs[b_end - 1] == ids[a2_start], (
                f"boundary at segment {i}->{i + 1}: labels[{b_end - 1}] must equal "
                f"ids[{a2_start}] (the first token of the next supervised segment)")


def test_partner_turns_are_never_supervised():
    """The model must learn to READ a partner turn, not produce one.

    SURVIVING MUTANT this test used to miss: the `supervised[t]`-instead-of-`[t+1]`
    bug leaks exactly ONE token per boundary -- the partner/prefix span's own FIRST
    token, used as the target of the supervised segment's last position. A search for
    a CONTIGUOUS RUN of an entire partner turn's ids can never see a one-token leak;
    it is built for a coarser failure that cannot occur here (nothing produces a whole
    partner turn as a training target) and is blind to the finer one that can. The
    fix: check every supervised target's OWN POSITION against every unsupervised
    span (prefix and both partner turns), not just whether it happens to start a
    string match of the whole turn.
    """
    skit = _skit()
    tok = _Tok()
    ex = build_skit_example(skit, tok, with_think=True, pad_token_id=0)
    ids, labs = ex["input_ids"], ex["labels"]

    # Old (contiguous-run) check, kept: a whole partner turn must never be the
    # supervised target sequence.
    supervised_targets = [ids[t + 1] for t, v in enumerate(labs) if v != -100]
    for p in PARTNER_TURNS:
        partner_ids = tok.encode(skit.turns[p], add_special_tokens=False)
        n = len(partner_ids)
        assert not any(supervised_targets[i:i + n] == partner_ids
                       for i in range(len(supervised_targets) - n + 1)), (
            f"partner turn {p} leaked into the supervised region")

    # New (per-position) check: no supervised target's POSITION may fall inside any
    # unsupervised span at all -- prefix or either partner turn -- catching a
    # single leaked token, not only a whole leaked turn.
    spans = _segment_spans(skit, tok, with_think=True)
    unsupervised_positions = {i for start, end, sup in spans if not sup
                              for i in range(start, end)}
    supervised_target_positions = {t + 1 for t, v in enumerate(labs) if v != -100}
    leaked = supervised_target_positions & unsupervised_positions
    assert not leaked, (
        f"supervised targets at positions {sorted(leaked)} point into an unsupervised "
        f"(prefix or partner) span -- a single-token leak, not a whole-turn one")


def test_every_example_is_tile_aligned():
    """ttml's SDPA backward mismatches raw-T against tile-padded-T and dies with TT_FATAL."""
    for arm in (True, False):
        ex = build_skit_example(_skit(), _Tok(), with_think=arm, pad_token_id=0)
        assert len(ex["input_ids"]) % TILE == 0
        assert len(ex["labels"]) == len(ex["input_ids"])


def test_think_blocks_appear_only_in_the_think_arm():
    """Mutation guard: a build that ignored with_think would pass a length-only check."""
    skit, tok = _skit(), _Tok()
    with_t = build_skit_example(skit, tok, with_think=True, pad_token_id=0)
    without = build_skit_example(skit, tok, with_think=False, pad_token_id=0)
    from train.improv import render_think
    block_ids = tok.encode(render_think(skit.blocks[0]), add_special_tokens=False)
    n = len(block_ids)

    def contains(hay, needle):
        return any(hay[i:i + len(needle)] == needle
                   for i in range(len(hay) - len(needle) + 1))

    assert contains(with_t["input_ids"], block_ids), "think arm must carry the block"
    assert not contains(without["input_ids"], block_ids), (
        "no-think arm must not leak the block")
