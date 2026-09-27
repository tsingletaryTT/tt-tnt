# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The shipped chat templates reproduce the training pipeline's tokens exactly.

The property under test is not "the template renders something plausible" but **the ids a
server sends for a chat are ids the model was trained on**. So every check here tokenizes
both sides with the real tokenizer and compares ids:

* rendered side: ``transformers.apply_chat_template`` (what vLLM calls) + ``tokenizer(...,
  add_special_tokens=False)`` (how vLLM tokenizes a chat prompt);
* training side: ``scripts/fetch_corpus.py::_render_dolly`` -> ``prepare_corpus.normalise`` +
  the ``</s>`` separator line -> ``train.tokenization.encode_batch`` line by line.

Fixture rows run everywhere the tokenizer artifact exists. The full-corpus sweep and the
search inside the real ``train_ids.npy`` streams are ``scripts/verify_chat_templates.py``
(results committed in ``docs/measurements/chat-template-proof.json``); a smaller slice of the
real corpus runs here when it is on disk.

Mutation-checked when written: replacing ``" Answer:"`` with ``"\\nAnswer:"`` in the
dolly_qa template, dropping the ``</s>``, or using a bare ``rstrip()`` in ``flat`` each turns
these red.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOKENIZER = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
pytestmark = pytest.mark.skipif(not TOKENIZER.is_file(), reason="no tokenizer artifact here")


@pytest.fixture(scope="module")
def tok():
    from scripts.verify_chat_templates import load_tokenizer

    return load_tokenizer()


def _dolly(instruction, response, context=""):
    from scripts.fetch_corpus import _render_dolly

    return _render_dolly({"instruction": instruction, "context": context, "response": response})


ROWS = [
    ("What is the capital of France?", "The capital of France is Paris.", ""),
    ("Which is a species of fish? Tope or Rope", "Tope", ""),
    ("When did Virgin Australia start operating?",
     "Virgin Australia commenced services on 31 August 2000 as Virgin Blue.",
     "Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an "
     "Australian-based airline.\n\nIt commenced services on 31 August 2000."),
    ("List three colors", "1. red\n2. green\n  3. indented blue", ""),
    # A line ENDING in U+202F (narrow no-break space), as real dolly doc 1541 does:
    # prepare_corpus.normalise keeps it (it strips only [ \t]), so the template must too --
    # a bare rstrip() drops it and the ids diverge. (The character between "here" and
    # "\\n" below is U+202F, not a plain space.)
    ("Trailing narrow space", "fine", "First line ends here \nSecond line."),
    ("Quote \"this\"", "He said \"no.\" She left.", ""),
    ("Narrow space and nbsp here?", "Kept as-is.", ""),
]


def _user(instruction, context):
    return instruction + (f"\n\n{context}" if context else "")


@pytest.mark.parametrize("i", range(len(ROWS)))
def test_dolly_qa_open_prompt_is_the_training_prefix(tok, i):
    """A single question renders to exactly the ids training saw before the answer."""
    from scripts.verify_chat_templates import pipeline_ids, render_ids

    q, a, ctx = ROWS[i]
    _, got = render_ids([{"role": "user", "content": _user(q, ctx)}], "dolly_qa", tok)
    full = pipeline_ids([_dolly(q, a, ctx)], tok)
    assert got == full[: len(got)], "open prompt must be a prefix of the trained document"
    # ...and what comes next in training is the answer, then </s>.
    assert tok.decode(full[len(got):]).strip().startswith(a.split("\n")[0].strip()[:10])
    assert full[-1] == 2


def test_dolly_qa_multi_turn_is_back_to_back_training_documents(tok):
    """History renders as closed dolly documents, each </s>-terminated, then an open one."""
    from scripts.verify_chat_templates import open_dolly, pipeline_ids, render_ids

    msgs, docs = [], []
    for q, a, ctx in ROWS[:3]:
        msgs += [{"role": "user", "content": _user(q, ctx)}, {"role": "assistant", "content": a}]
        docs.append(_dolly(q, a, ctx))
    q, _, ctx = ROWS[3]
    msgs = msgs[-4:] + [{"role": "user", "content": _user(q, ctx)}]  # inside the 5-message cap
    _, got = render_ids(msgs, "dolly_qa", tok)
    want = pipeline_ids(docs[-2:], tok) + pipeline_ids([open_dolly(_user(q, ctx))], tok,
                                                       terminate=False)
    assert got == want


def test_dolly_qa_reply_with_vllm_leading_space_rerenders_identically(tok):
    """vLLM returns content as ' Paris' (the generated tokens); it must re-render as trained."""
    from scripts.verify_chat_templates import render_ids

    base = [{"role": "user", "content": "Q?"}]
    _, a = render_ids(base + [{"role": "assistant", "content": " Paris."}], "dolly_qa", tok)
    _, b = render_ids(base + [{"role": "assistant", "content": "Paris."}], "dolly_qa", tok)
    assert a == b


def test_dolly_qa_never_emits_a_newline_token(tok):
    from scripts.verify_chat_templates import render_ids

    nl = {i for t, i in tok.get_vocab().items() if "Ċ" in t or "č" in t}
    msgs = [{"role": "user", "content": _user(*ROWS[2][::2])},
            {"role": "assistant", "content": "a\nb\n\nc"}, {"role": "user", "content": "x\ny"}]
    for fmt in ("dolly_qa", "plain"):
        _, ids = render_ids(msgs, fmt, tok)
        assert not nl & set(ids), fmt


def test_system_messages_are_dropped(tok):
    from scripts.verify_chat_templates import render_ids

    u = [{"role": "user", "content": "Hi?"}]
    for fmt in ("dolly_qa", "plain"):
        with_sys = render_ids([{"role": "system", "content": "Be terse."}] + u, fmt, tok)
        assert with_sys == render_ids(u, fmt, tok), fmt


def test_plain_is_the_story_itself(tok):
    """plain: an opening + a continuation render as the one document training saw."""
    from scripts.verify_chat_templates import pipeline_ids, render_ids

    story = ("One day, a little girl named Lily found a needle in her room.\n"
             "She knew it was difficult to play with it because it was sharp.\n\n"
             "\"Mom, I found this needle,\" she said.")
    lines = story.split("\n")
    msgs = [{"role": "user", "content": lines[0]},
            {"role": "assistant", "content": " " + "\n".join(lines[1:])}]
    _, got = render_ids(msgs, "plain", tok)
    assert got == pipeline_ids([story], tok, terminate=False)
    assert 2 not in got, "plain continuation must not insert a document separator"


def test_real_corpus_slice_when_present(tok):
    """First 500 real dolly documents / TinyStories stories, when the corpus is on disk."""
    from scripts.verify_chat_templates import (DIALOGUE, TINYSTORIES, check_dolly, check_plain,
                                               read_documents)

    if not DIALOGUE.is_file() or not TINYSTORIES.is_file():
        pytest.skip("prepared corpus not on this machine")
    d = check_dolly(tok, read_documents(DIALOGUE, 500))
    assert d["identical"] == d["pairs_checked"] > 0, d["first_mismatches"]
    assert d["legacy_tool_call_sft_identical"] == 0, "the old template never matched pretraining"
    p = check_plain(tok, read_documents(TINYSTORIES, 500))
    assert p["identical"] == p["docs_checked"] > 0, p["first_mismatches"]


def test_committed_proof_covers_the_whole_corpus():
    """The committed evidence file says what the docs claim it says."""
    proof = json.loads((ROOT / "docs" / "measurements" / "chat-template-proof.json").read_text())
    assert proof["dolly_qa"]["identical"] == proof["dolly_qa"]["pairs_checked"] == 15006
    assert proof["plain"]["identical"] == proof["plain"]["docs_checked"]
    assert proof["dolly_qa_stream"]["found_at_offset"] is not None
    assert proof["dolly_qa_stream"]["legacy_tool_call_sft_found_at_offset"] is None
    assert proof["plain_stream"]["found_at_offset"] is not None


def test_publish_targets_name_the_measured_formats():
    from convert.chat_templates import CORPUS_CHAT_FORMATS
    from scripts.publish_to_hub import TARGETS

    assert TARGETS["episod/tt-tnt"]["chat_format"] == CORPUS_CHAT_FORMATS["tokens-v3"] == "plain"
    assert TARGETS["episod/tt-tnt-1024"]["chat_format"] == CORPUS_CHAT_FORMATS["tokens-v4"]


def test_infer_chat_format_from_real_headers():
    from convert.chat_templates import infer_chat_format

    assert infer_chat_format({"tokens_dir": "artifacts/tokens-v4"}) == "dolly_qa"
    # format-1 header (tt-tnt-v3's own): no tokens_dir, exact corpus total.
    assert infer_chat_format({"corpus_tokens": 391921555}) == "plain"
    assert infer_chat_format({"tokens_dir": "artifacts/tokens-stagea"}) is None
    assert infer_chat_format({}) is None


def test_publish_guard_refuses_an_artifact_with_the_wrong_template(tmp_path):
    """The upload-side guard: an artifact carrying the OLD template (or none) cannot ship."""
    from convert.chat_templates import TOOL_CALL_SFT_TEMPLATE, template_for
    from scripts.publish_to_hub import _assert_local_artifact_is_publishable, target_for

    target = target_for("episod/tt-tnt-1024")
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 512}))
    cfg = tmp_path / "tokenizer_config.json"
    for bad in ({"chat_template": TOOL_CALL_SFT_TEMPLATE}, {}):
        cfg.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="dolly_qa"):
            _assert_local_artifact_is_publishable(tmp_path, target)
    cfg.write_text(json.dumps({"chat_template": template_for("dolly_qa")}))
    _assert_local_artifact_is_publishable(tmp_path, target)  # the right one passes
