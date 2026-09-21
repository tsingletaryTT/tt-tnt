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
