import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_storycloze import StoryClozeItem  # noqa: E402


def test_storycloze_item_rejects_invalid_correct_ending():
    with pytest.raises(ValueError, match="correct_ending must be 1 or 2"):
        StoryClozeItem(
            story_id="x", context_sentences=("a", "b", "c", "d"),
            ending_1="e1", ending_2="e2", correct_ending=3,
        )


def test_storycloze_item_accepts_valid_construction():
    item = StoryClozeItem(
        story_id="x", context_sentences=("a", "b", "c", "d"),
        ending_1="e1", ending_2="e2", correct_ending=1,
    )
    assert item.correct_ending == 1
    assert item.context_sentences == ("a", "b", "c", "d")


def test_load_storycloze_items_train_split_has_plausible_shape():
    """Requires network access to the HF Hub -- skips cleanly without it."""
    from scripts.eval_storycloze import load_storycloze_items

    try:
        items = load_storycloze_items("train")
    except Exception as exc:  # noqa: BLE001 - network/environment, not a code defect
        pytest.skip(f"could not reach the HF Hub: {exc}")
    assert len(items) > 0
    first = items[0]
    assert isinstance(first.story_id, str) and first.story_id
    assert all(isinstance(s, str) and s for s in first.context_sentences)
    assert first.correct_ending in (1, 2)


def _tiny_causal_lm_and_tokenizer():
    """A tiny randomly-initialized GPT2-architecture model+tokenizer for fast CPU tests.

    Not this project's own model -- the scorer is architecture-agnostic (works on any
    AutoModelForCausalLM), and using a tiny stock model keeps these tests independent of
    artifacts/hf-tt-tnt-* being present on disk.
    """
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    # add_prefix_space=True matches this project's own trained tokenizer's behavior (see
    # CLAUDE.md's tokenizer-and-corpus entry): tokenize_sentence()'s per-sentence encoding
    # relies on every encode call injecting a leading space so a sentence tokenized standalone
    # matches how it tokenizes mid-string. GPT2's stock tokenizer does NOT do this by default
    # -- confirmed directly: without add_prefix_space=True, tokenizing "She walked back home."
    # standalone (leading token "She", id 3347) disagrees with how the same words tokenize
    # inside a longer string (leading token "Ġshe", id 1375), which silently corrupts the
    # not-hollow test below (the fine-tuned model never actually saw the token sequence being
    # scored). Setting it here makes the fixture's tokenizer behave like production's.
    tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
    config = GPT2Config(vocab_size=tokenizer.vocab_size, n_embd=32, n_layer=2, n_head=2,
                        n_positions=128)
    model = GPT2LMHeadModel(config).eval()
    return model, tokenizer


def test_score_ending_produces_expected_token_count():
    from scripts.eval_storycloze import score_ending, tokenize_sentence

    model, tokenizer = _tiny_causal_lm_and_tokenizer()
    context_ids = tokenize_sentence(tokenizer, "The sky is blue.")
    ending_ids = tokenize_sentence(tokenizer, "It rained.")
    result = score_ending(model, context_ids, ending_ids)
    assert result.n_tokens == len(ending_ids)
    assert isinstance(result.raw_sum_logprob, float)
    assert isinstance(result.mean_logprob, float)
    # mean is the sum averaged over exactly n_tokens, not some other count
    assert result.mean_logprob == pytest.approx(
        result.raw_sum_logprob / result.n_tokens, rel=1e-6)


def test_score_ending_requires_at_least_one_context_and_one_ending_token():
    from scripts.eval_storycloze import score_ending

    model, tokenizer = _tiny_causal_lm_and_tokenizer()
    with pytest.raises(ValueError, match="at least 2 tokens"):
        score_ending(model, context_ids=[], ending_ids=[])


def _tiny_causal_lm_finetuned_on(tokenizer, text: str, steps: int = 30):
    import torch

    model, _ = _tiny_causal_lm_and_tokenizer()
    ids = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        logits = model(ids[:, :-1]).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
    model.eval()
    return model


def test_scorer_prefers_the_real_continuation_over_a_garbled_one():
    """The not-hollow proof: the scorer must be able to discriminate at all.

    A garbled ending (its own tokens reversed) is not a naturally-occurring completion of
    the context, so a working scorer must give it lower likelihood than the real, coherent
    continuation. This is checked BEFORE trusting the scorer on real Story Cloze data.
    """
    from scripts.eval_storycloze import score_ending, tokenize_sentence

    _, tokenizer = _tiny_causal_lm_and_tokenizer()
    context_ids = tokenize_sentence(
        tokenizer, "Sarah walked to the store. She bought some bread. She paid with cash.")
    real_ids = tokenize_sentence(tokenizer, "She walked back home.")
    garbled_ids = list(reversed(real_ids))

    # NOTE: a randomly-initialized tiny model has no learned preference, so this specific
    # assertion is checked against a model that has SEEN this exact continuation during a
    # few steps of fine-tuning within the test -- see the fixture above, not the bare
    # untrained model from _tiny_causal_lm_and_tokenizer().
    model = _tiny_causal_lm_finetuned_on(
        tokenizer,
        "Sarah walked to the store. She bought some bread. She paid with cash. "
        "She walked back home.")

    real_score = score_ending(model, context_ids, real_ids)
    garbled_score = score_ending(model, context_ids, garbled_ids)

    assert real_score.mean_logprob > garbled_score.mean_logprob


def test_score_item_returns_both_endings_full_and_blind():
    from scripts.eval_storycloze import StoryClozeItem, score_item

    model, tokenizer = _tiny_causal_lm_and_tokenizer()
    item = StoryClozeItem(
        story_id="s1", context_sentences=("A.", "B.", "C.", "D."),
        ending_1="It ended well.", ending_2="It ended badly.", correct_ending=1)
    result = score_item(model, tokenizer, item)
    assert result.story_id == "s1"
    assert result.correct_ending == 1
    for field_name in ("full_1", "full_2", "blind_1", "blind_2"):
        score = getattr(result, field_name)
        assert score.n_tokens > 0


def test_context_blind_scores_are_identical_regardless_of_context():
    """The context-blind control must be provably blind, not merely named blind.

    Two DIFFERENT contexts, same ending text: if the "blind" score ever changes with the
    context, the control is leaking context and is not measuring what it claims to measure.
    """
    from scripts.eval_storycloze import StoryClozeItem, score_item

    model, tokenizer = _tiny_causal_lm_and_tokenizer()
    item_a = StoryClozeItem(
        story_id="a", context_sentences=("The sun rose.", "Birds sang.", "It was warm.", "Dew glistened."),
        ending_1="She smiled.", ending_2="unused", correct_ending=1)
    item_b = StoryClozeItem(
        story_id="b", context_sentences=("The city burned.", "Sirens wailed.", "Smoke rose.", "People fled."),
        ending_1="She smiled.", ending_2="unused", correct_ending=1)

    result_a = score_item(model, tokenizer, item_a)
    result_b = score_item(model, tokenizer, item_b)

    assert result_a.blind_1.raw_sum_logprob == pytest.approx(result_b.blind_1.raw_sum_logprob)
    assert result_a.blind_1.mean_logprob == pytest.approx(result_b.blind_1.mean_logprob)


def test_aggregate_computes_accuracy_and_length_bias_per_normalization():
    from scripts.eval_storycloze import (
        EndingScore, ItemResult, aggregate,
    )

    # Item 1: correct ending (1) scores higher on both raw and mean -- a clean correct case.
    # Item 2: correct ending (2) scores LOWER on raw sum (because it's longer) but higher on
    #   mean-per-token -- this is the length-bias case the aggregation must be able to show.
    results = [
        ItemResult(
            story_id="i1", correct_ending=1,
            full_1=EndingScore(raw_sum_logprob=-2.0, mean_logprob=-1.0, n_tokens=2),
            full_2=EndingScore(raw_sum_logprob=-5.0, mean_logprob=-2.5, n_tokens=2),
            blind_1=EndingScore(raw_sum_logprob=-2.0, mean_logprob=-1.0, n_tokens=2),
            blind_2=EndingScore(raw_sum_logprob=-2.0, mean_logprob=-1.0, n_tokens=2),
        ),
        ItemResult(
            story_id="i2", correct_ending=2,
            full_1=EndingScore(raw_sum_logprob=-3.0, mean_logprob=-3.0, n_tokens=1),
            full_2=EndingScore(raw_sum_logprob=-4.0, mean_logprob=-1.0, n_tokens=4),
            blind_1=EndingScore(raw_sum_logprob=-1.0, mean_logprob=-1.0, n_tokens=1),
            blind_2=EndingScore(raw_sum_logprob=-1.0, mean_logprob=-1.0, n_tokens=1),
        ),
    ]
    stats = aggregate(results)
    assert set(stats) == {"raw_sum", "mean_per_token"}
    # raw_sum: item 1 correct (picks ending 1, higher raw), item 2 WRONG (picks ending 1,
    # -3.0 > -4.0, but correct is ending 2) -> 1/2 accuracy
    assert stats["raw_sum"].accuracy == pytest.approx(0.5)
    # mean_per_token: item 1 correct, item 2 correct (-1.0 > -3.0, picks ending 2) -> 2/2
    assert stats["mean_per_token"].accuracy == pytest.approx(1.0)
    assert stats["raw_sum"].n_items == 2
    assert 0 <= stats["raw_sum"].class_balance_fraction_answer_1 <= 1
    assert stats["raw_sum"].length_bias_correlation is not None


def test_choose_headline_normalization_prefers_lower_absolute_length_bias():
    from scripts.eval_storycloze import NormalizationStats, choose_headline_normalization

    stats = {
        "raw_sum": NormalizationStats(
            n_items=10, accuracy=0.6, context_blind_accuracy=0.55,
            class_balance_fraction_answer_1=0.5,
            length_bias_correlation=0.7, context_blind_length_bias_correlation=0.6),
        "mean_per_token": NormalizationStats(
            n_items=10, accuracy=0.65, context_blind_accuracy=0.52,
            class_balance_fraction_answer_1=0.5,
            length_bias_correlation=-0.1, context_blind_length_bias_correlation=0.05),
    }
    assert choose_headline_normalization(stats) == "mean_per_token"


def test_choose_headline_normalization_treats_none_bias_as_worst_case():
    """A normalization whose length-bias could not even be computed (zero variance) is not
    silently preferred just because None fails a naive comparison in its favor."""
    from scripts.eval_storycloze import NormalizationStats, choose_headline_normalization

    stats = {
        "raw_sum": NormalizationStats(
            n_items=10, accuracy=0.6, context_blind_accuracy=0.55,
            class_balance_fraction_answer_1=0.5,
            length_bias_correlation=None, context_blind_length_bias_correlation=None),
        "mean_per_token": NormalizationStats(
            n_items=10, accuracy=0.65, context_blind_accuracy=0.52,
            class_balance_fraction_answer_1=0.5,
            length_bias_correlation=0.2, context_blind_length_bias_correlation=0.05),
    }
    assert choose_headline_normalization(stats) == "mean_per_token"


def _minimal_report(revision: str, split: str, per_item: list, headline_norm: str = "mean_per_token") -> dict:
    return {
        "schema": "tt-tnt/storycloze/1",
        "dataset_revision": revision,
        "split": split,
        "headline_normalization": headline_norm,
        "per_item": per_item,
    }


def test_compare_reports_runs_a_paired_sign_test_over_headline_correctness():
    from scripts.eval_storycloze import compare_reports

    # Both models score item 1 correctly (concordant), item 2: model A wrong, model B right
    # (discordant, favors B), item 3: model A right, model B wrong (discordant, favors A).
    per_item_a = [
        {"story_id": "s1", "correct_ending": 1, "chosen_mean_per_token": 1},
        {"story_id": "s2", "correct_ending": 1, "chosen_mean_per_token": 2},
        {"story_id": "s3", "correct_ending": 1, "chosen_mean_per_token": 1},
    ]
    per_item_b = [
        {"story_id": "s1", "correct_ending": 1, "chosen_mean_per_token": 1},
        {"story_id": "s2", "correct_ending": 1, "chosen_mean_per_token": 1},
        {"story_id": "s3", "correct_ending": 1, "chosen_mean_per_token": 2},
    ]
    a = _minimal_report("rev1", "eval", per_item_a)
    b = _minimal_report("rev1", "eval", per_item_b)

    result = compare_reports(a, b)
    assert result["n_items_compared"] == 3
    assert result["sign_test"]["n"] == 2  # two discordant pairs
    assert result["sign_test"]["n_negative"] == 1  # favors A (item s3)
    assert result["sign_test"]["n_positive"] == 1  # favors B (item s2)


def test_compare_reports_refuses_mismatched_dataset_revision():
    from scripts.eval_storycloze import compare_reports

    a = _minimal_report("rev1", "eval", [])
    b = _minimal_report("rev2", "eval", [])
    with pytest.raises(ValueError, match="dataset_revision"):
        compare_reports(a, b)


def test_compare_reports_refuses_mismatched_split():
    from scripts.eval_storycloze import compare_reports

    a = _minimal_report("rev1", "eval", [])
    b = _minimal_report("rev1", "train", [])
    with pytest.raises(ValueError, match="split"):
        compare_reports(a, b)


def test_compare_reports_refuses_mismatched_headline_normalization():
    """A and B must have SELECTED the same normalization, not merely be scorable under one.

    ``compare_reports`` reads A's ``headline_normalization`` and applies it to both reports.
    Before this refusal, a B whose own run chose ``raw_sum`` would have been silently judged
    by ``mean_per_token`` with nothing in the output recording that substitution.
    """
    from scripts.eval_storycloze import compare_reports

    a = _minimal_report("rev1", "eval", [], headline_norm="mean_per_token")
    b = _minimal_report("rev1", "eval", [], headline_norm="raw_sum")
    with pytest.raises(ValueError, match="headline_normalization"):
        compare_reports(a, b)


def test_build_report_refuses_items_and_results_of_different_lengths(tmp_path):
    """``zip`` would truncate silently; the report would be short with no error raised."""
    from scripts.eval_storycloze import StoryClozeItem, ItemResult, EndingScore, build_report

    items = [
        StoryClozeItem(story_id=f"s{i}", context_sentences=("A.", "B.", "C.", "D."),
                       ending_1="ok", ending_2="bad", correct_ending=1)
        for i in range(2)
    ]
    result = ItemResult(
        story_id="s0", correct_ending=1,
        full_1=EndingScore(raw_sum_logprob=-1.0, mean_logprob=-0.5, n_tokens=2),
        full_2=EndingScore(raw_sum_logprob=-4.0, mean_logprob=-2.0, n_tokens=2),
        blind_1=EndingScore(raw_sum_logprob=-3.0, mean_logprob=-1.5, n_tokens=2),
        blind_2=EndingScore(raw_sum_logprob=-3.0, mean_logprob=-1.5, n_tokens=2),
    )
    model_dir = tmp_path / "hf-fake"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"fake weights")

    with pytest.raises(ValueError, match="same length, got 2 and 1"):
        build_report(model_dir, items, [result], split="eval", revision="rev1")


def test_pick_ending_is_the_single_rule_both_call_sites_use(tmp_path):
    """The stored ``chosen_*`` fields and the accuracy count must come from one definition.

    Monkeypatching ``pick_ending`` to the inverted rule has to move BOTH the per-item
    ``chosen_*`` values and ``aggregate``'s accuracy -- if either kept an inline copy of the
    rule, one of these assertions fails.
    """
    import scripts.eval_storycloze as mod

    assert mod.pick_ending(-1.0, -2.0) == 1
    assert mod.pick_ending(-2.0, -1.0) == 2
    assert mod.pick_ending(-1.0, -1.0) == 2  # tie goes to ending 2

    item = mod.StoryClozeItem(
        story_id="s1", context_sentences=("A.", "B.", "C.", "D."),
        ending_1="ok", ending_2="bad", correct_ending=1)
    result = mod.ItemResult(
        story_id="s1", correct_ending=1,
        full_1=mod.EndingScore(raw_sum_logprob=-1.0, mean_logprob=-0.5, n_tokens=2),
        full_2=mod.EndingScore(raw_sum_logprob=-4.0, mean_logprob=-2.0, n_tokens=2),
        blind_1=mod.EndingScore(raw_sum_logprob=-3.0, mean_logprob=-1.5, n_tokens=2),
        blind_2=mod.EndingScore(raw_sum_logprob=-3.0, mean_logprob=-1.5, n_tokens=2),
    )
    model_dir = tmp_path / "hf-fake"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"fake weights")

    baseline = mod.build_report(model_dir, [item], [result], split="eval", revision="rev1")
    assert baseline["per_item"][0]["chosen_mean_per_token"] == 1
    assert baseline["normalizations"]["mean_per_token"]["accuracy"] == pytest.approx(1.0)

    original = mod.pick_ending
    try:
        mod.pick_ending = lambda s1, s2: 2 if s1 > s2 else 1
        inverted = mod.build_report(model_dir, [item], [result], split="eval", revision="rev1")
    finally:
        mod.pick_ending = original

    assert inverted["per_item"][0]["chosen_mean_per_token"] == 2
    assert inverted["normalizations"]["mean_per_token"]["accuracy"] == pytest.approx(0.0)


def test_build_report_records_provenance_and_per_item_scores(tmp_path):
    from scripts.eval_storycloze import StoryClozeItem, ItemResult, EndingScore, build_report

    item = StoryClozeItem(
        story_id="s1", context_sentences=("A.", "B.", "C.", "D."),
        ending_1="ok", ending_2="bad", correct_ending=1)
    result = ItemResult(
        story_id="s1", correct_ending=1,
        full_1=EndingScore(raw_sum_logprob=-1.0, mean_logprob=-0.5, n_tokens=2),
        full_2=EndingScore(raw_sum_logprob=-4.0, mean_logprob=-2.0, n_tokens=2),
        blind_1=EndingScore(raw_sum_logprob=-3.0, mean_logprob=-1.5, n_tokens=2),
        blind_2=EndingScore(raw_sum_logprob=-3.0, mean_logprob=-1.5, n_tokens=2),
    )
    model_dir = tmp_path / "hf-fake"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"fake weights")

    report = build_report(model_dir, [item], [result], split="eval", revision="rev1")

    assert report["schema"] == "tt-tnt/storycloze/1"
    assert report["dataset_revision"] == "rev1"
    assert report["split"] == "eval"
    assert report["model_dir"] == str(model_dir)
    assert len(report["model_weights_sha256"]) == 64  # hex sha256 digest length
    assert report["headline_normalization"] in ("raw_sum", "mean_per_token")
    assert len(report["per_item"]) == 1
    row = report["per_item"][0]
    assert row["story_id"] == "s1"
    assert row["correct_ending"] == 1
    assert "chosen_raw_sum" in row and "chosen_mean_per_token" in row


def test_rescore_from_report_reproduces_the_original_aggregate_stats(tmp_path):
    from scripts.eval_storycloze import (
        StoryClozeItem, ItemResult, EndingScore, build_report, rescore_from_report,
    )

    item = StoryClozeItem(
        story_id="s1", context_sentences=("A.", "B.", "C.", "D."),
        ending_1="ok", ending_2="bad", correct_ending=1)
    result = ItemResult(
        story_id="s1", correct_ending=1,
        full_1=EndingScore(raw_sum_logprob=-1.0, mean_logprob=-0.5, n_tokens=2),
        full_2=EndingScore(raw_sum_logprob=-4.0, mean_logprob=-2.0, n_tokens=2),
        blind_1=EndingScore(raw_sum_logprob=-3.0, mean_logprob=-1.5, n_tokens=2),
        blind_2=EndingScore(raw_sum_logprob=-3.0, mean_logprob=-1.5, n_tokens=2),
    )
    model_dir = tmp_path / "hf-fake"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"fake weights")

    original = build_report(model_dir, [item], [result], split="eval", revision="rev1")
    rescored = rescore_from_report(original)

    assert rescored["normalizations"] == original["normalizations"]
    assert rescored["headline_normalization"] == original["headline_normalization"]


def test_rescore_from_cli_needs_no_torch_or_transformers_import(tmp_path):
    """Verified in a subprocess, matching tests/test_ttml_forward.py's import-purity pattern --
    this test session may have already imported torch/transformers elsewhere."""
    import json
    import subprocess

    report = {
        "schema": "tt-tnt/storycloze/1", "dataset_revision": "rev1", "split": "eval",
        "dataset": "juletxara/xstory_cloze", "dataset_config": "en",
        "model_dir": "fake", "model_weights_sha256": "0" * 64,
        "normalizations": {}, "headline_normalization": "mean_per_token",
        "headline_accuracy": 0.0,
        "per_item": [{
            "story_id": "s1", "correct_ending": 1,
            "full_1_raw_sum": -1.0, "full_2_raw_sum": -4.0,
            "full_1_mean": -0.5, "full_2_mean": -2.0,
            "full_1_n_tokens": 2, "full_2_n_tokens": 2,
            "blind_1_raw_sum": -3.0, "blind_2_raw_sum": -3.0,
            "blind_1_mean": -1.5, "blind_2_mean": -1.5,
            "chosen_raw_sum": 1, "chosen_mean_per_token": 1,
        }],
    }
    in_path = tmp_path / "in.json"
    out_path = tmp_path / "out.json"
    in_path.write_text(json.dumps(report))

    probe = (
        "import sys; "
        "from scripts.eval_storycloze import main; "
        f"main(['--rescore-from', {str(in_path)!r}, '--out', {str(out_path)!r}]); "
        "bad = [m for m in ('torch', 'transformers') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                           check=True, cwd=str(ROOT))
    # NOTE: deliberately not .strip()'d before .splitlines() -- main()'s own
    # "[rescore] wrote ..." print precedes the probe's print(','.join(bad)) line, and when
    # bad is empty that second print emits a blank line; stripping the whole string first
    # would eat that trailing blank line and silently compare against the wrong line.
    assert result.stdout.splitlines()[-1] == "", (
        f"--rescore-from pulled in: {result.stdout.strip()}")
    assert out_path.is_file()


def test_eval_storycloze_module_imports_no_tenstorrent():
    """Matches tests/test_ttml_forward.py's pattern: checked in a subprocess since this test
    session may already have imported plenty of things transitively."""
    import subprocess

    probe = (
        "import sys; import scripts.eval_storycloze; "
        "bad=[m for m in ('ttnn','ttml') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                        check=True, cwd=str(ROOT))
    assert out.stdout.strip() == "", f"scripts.eval_storycloze pulled in: {out.stdout.strip()}"


def test_readme_documents_storycloze_license():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "juletxara/xstory_cloze" in readme
    assert "CC BY-SA 4.0" in readme
