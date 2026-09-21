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
