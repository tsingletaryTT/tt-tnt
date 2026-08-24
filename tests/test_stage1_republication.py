# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The stage-1 republication, tested at the PUBLICATION layer.

Why this file exists. The splitter cutover changed a number that is already published, and
the spec's answer was republication rather than a silent overwrite. A note in a JSON file is
worth nothing if it drifts from the code, so these tests do two things a comment cannot:

  1. re-derive the published numbers from the REAL corpus and require an EXACT match. The
     republished `kept`/`drop_rate` is therefore falsifiable: change the splitter again and
     this file goes red until the artifact is updated.
  2. require the OLD numbers to still be present and unchanged. "Beside the old ones" is
     the actual instruction, and an overwrite that merely added the new block would satisfy
     a laxer test.

Requirement 1 of the spec's testing section -- "nothing decides a published claim from
inside main()" -- is why the numbers here come from `derive_from_story` and
`derive_skit`, both callable, rather than from re-running a script and scraping stdout.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import ROOT, needs_artifacts  # noqa: E402

import scripts.derive_traces as derive_traces  # noqa: E402
from scripts.derive_dialogue_skits import iter_stories  # noqa: E402
from scripts.derive_skits import build_idf  # noqa: E402
from scripts.score_improv import intensity, load_harm_lexicon  # noqa: E402
from train.improv import split_sentences  # noqa: E402
from train.skit import derive_skit  # noqa: E402

CORPUS = "artifacts/corpus/tinystories.txt"
STAGE1 = ROOT / "docs" / "measurements" / "improv-stage1.json"
STAGE2 = ROOT / "docs" / "measurements" / "skits-stage2.json"

#: The published run's parameters. Not knobs -- changing either invalidates the comparison.
N_STORIES = 20000
SEED = 5489


@pytest.fixture(scope="module")
def stage1():
    return json.loads(STAGE1.read_text())


@pytest.fixture(scope="module")
def stage2():
    return json.loads(STAGE2.read_text())


def test_the_old_derivation_is_still_there_unchanged(stage1):
    """"The new numbers BESIDE the old ones." The old block is the historical record of
    what the published checkpoints were trained on and must not be edited."""
    old = stage1["derivation"]
    assert old["kept"] == 18791
    assert old["drop_rate"] == 0.0605
    assert old["stories"] == N_STORIES and old["seed"] == SEED
    assert old["drops_by_rule"] == {"no_carry_or_no_add": 1209}


def test_the_file_says_what_is_superseded_and_what_is_not(stage1):
    """A `superseded_by` that does not say the RATES were never re-measured would be the
    more misleading of the two possible errors."""
    s = stage1["superseded_by"]
    assert "derivation_republished_2026_08_23" in s["superseded_by"]
    assert "2026-08-23-reach-dial-design.md" in s["approved_by"]
    assert "re-MEASURED" in s["what_is_NOT_superseded"]
    assert "artifacts/improv/traces.jsonl" in s["old_artifacts_preserved"]
    # The rates themselves are untouched, and stay untouched.
    assert stage1["rates"]["groundedness"]["t"] == 0.232
    assert stage1["success_criteria"]["all_criteria_met"] is False


@needs_artifacts(CORPUS, reason="the republished kept/drop numbers are corpus properties")
def test_the_republished_numbers_are_reproducible_from_the_corpus(stage1):
    """Re-derive stage 1 from the real corpus and require the published values EXACTLY.

    This is the test that makes the republication a measurement rather than an assertion.
    `derive_traces.DROPS` is a module-level Counter, so it is reset here; leaving it dirty
    would silently accumulate across tests and inflate the drop table.
    """
    rep = stage1["derivation_republished_2026_08_23"]
    stories = list(iter_stories(ROOT / CORPUS, N_STORIES))
    assert len(stories) == N_STORIES
    idf = derive_traces.build_idf(stories)

    saved, fresh = derive_traces.DROPS, Counter()
    derive_traces.DROPS = fresh          # module-global; derive_from_story reads it by name
    kept = 0
    for i, story in enumerate(stories):
        if derive_traces.derive_from_story(story, story_id=i, rng_seed=SEED,
                                           idf=idf) is not None:
            kept += 1

    derive_traces.DROPS = saved          # never leave module state dirty for other files

    assert kept == rep["new"]["kept"] == 18970, kept
    assert round(1 - kept / N_STORIES, 4) == rep["new"]["drop_rate"] == 0.0515
    assert dict(fresh) == rep["new"]["drops_by_rule"] == {"no_carry_or_no_add": 1025,
                                                          "too_few_sentences": 5}
    # ...and the delta the file reports is the delta between the two blocks it publishes.
    assert rep["new"]["kept"] - rep["old"]["kept"] == rep["delta"]["kept"] == 179


def test_the_republication_admits_the_examples_themselves_changed(stage1):
    """The drop rate moved by 0.9pp; 40% of the shared examples changed TEXT.

    A republication that reported only the drop rate would read as a rounding correction.
    """
    churn = stage1["derivation_republished_2026_08_23"]["churn_is_not_small"]
    assert churn["same_prefix_and_continuation_fraction"] < 0.7
    assert churn["same_cut_k_fraction"] < 0.8
    assert "would not be a small perturbation" in churn["reading"]


@needs_artifacts(CORPUS, reason="dialogue density is a corpus property")
def test_the_splitter_moved_the_skit_population_from_dialogue_poor_to_dialogue_rich(stage2):
    """The claim in stage 2's note, re-measured on the real corpus.

    Measured over 4,000 stories rather than the published 20,000 so the suite stays fast;
    the assertion is therefore on the DIRECTION and the side of 1.0, which is the claim that
    matters. The exact published figures come from the 20,000-story run recorded in the file
    and are consistency-checked against each other below.
    """
    note = stage2["splitter_superseded_2026_08_23"]["measured"]
    assert note["old_turn_dialogue_ratio_vs_corpus"] < 1.0 < \
           note["new_turn_dialogue_ratio_vs_corpus"]
    assert note["new_kept_fraction"] > note["old_kept_fraction"]
    assert note["n_stories"] == N_STORIES

    harm = load_harm_lexicon()
    stories = list(iter_stories(ROOT / CORPUS, 4000))
    idf = build_idf(stories)
    units = quoted_units = turns = dialogue_turns = kept = 0
    for i, story in enumerate(stories):
        us = split_sentences(story)
        units += len(us)
        quoted_units += sum(1 for u in us if '"' in u)
        skit = derive_skit(story, story_id=i, idf=idf,
                           intensity=lambda t: intensity(t, harm))
        if skit is None:
            continue
        kept += 1
        turns += len(skit.turns)
        dialogue_turns += sum(1 for t in skit.turns if '"' in t)

    ratio = (dialogue_turns / turns) / (quoted_units / units)
    assert ratio > 1.0, f"skit turns are dialogue-POOR again: ratio {ratio:.3f}"
    assert kept / len(stories) > 0.10, kept / len(stories)


def test_stage2s_note_does_not_claim_the_results_were_rerun(stage2):
    n = stage2["splitter_superseded_2026_08_23"]
    assert "NOT reproducible from HEAD" in n["consequence"]
    assert "No result in this file has been re-run" in n["not_re_measured"]
    # The original verdict is untouched.
    assert "verdict" in stage2
