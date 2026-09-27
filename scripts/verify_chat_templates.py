#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Prove each shipped chat template reproduces its model's training tokens exactly.

"Byte-for-byte" for a chat template means TOKEN-for-token: the ids a server feeds the model
for a rendered conversation must be the ids the training pipeline produced for the same
text. This script checks that three ways, from weakest to strongest, and writes the
evidence to ``docs/measurements/chat-template-proof.json``:

1. **Pipeline diff over the whole dolly slice** (``dolly_qa``). Every document in
   ``artifacts/corpus/dialogue.txt`` (the prepared dolly-15k slice tokens-v4 was built
   from) is split back into (question, answer); consecutive pairs are rendered as a chat
   ``[user_i, assistant_i, user_i+1]`` through ``transformers.apply_chat_template`` -- the
   same call vLLM makes -- and tokenized the way vLLM tokenizes a chat prompt
   (``add_special_tokens=False``). That is compared with the pipeline's own encoding
   (``prepare_corpus.normalise`` + the ``</s>`` separator line + ``train.tokenization.
   encode_batch`` line by line) of doc_i followed by the open prefix of doc_i+1.
2. **The same for ``plain``** over TinyStories documents (``artifacts/corpus/tinystories.txt``):
   a story split into a user opening and an assistant continuation.
3. **Found verbatim in the real training stream.** The rendered ids of a multi-turn
   conversation are searched for in ``artifacts/tokens-v4/train_ids.npy`` (``dolly_qa``) /
   ``artifacts/tokens-v3/train_ids.npy`` (``plain``). A hit means the exact id sequence the
   server will send is one the model was trained on, byte for byte.

It also renders the same conversation through the legacy ``tool_call_sft`` template (what
tt-tnt-1024 shipped until 2026-09-27) and records how far that is from the training ids.

No device, no ttnn: tokenizer + numpy only. Run from the repo root::

    python scripts/verify_chat_templates.py            # full run, writes the JSON
    python scripts/verify_chat_templates.py --limit 500 --no-stream
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from convert.chat_templates import CHAT_TEMPLATES  # noqa: E402
from scripts.prepare_corpus import DOCUMENT_SEPARATOR, normalise  # noqa: E402
from train.tokenization import encode_batch  # noqa: E402

TOKENIZER_DIR = ROOT / "artifacts" / "tokenizer"
DIALOGUE = ROOT / "artifacts" / "corpus" / "dialogue.txt"
TINYSTORIES = ROOT / "artifacts" / "corpus" / "tinystories.txt"
STREAMS = {"dolly_qa": ROOT / "artifacts" / "tokens-v4" / "train_ids.npy",
           "plain": ROOT / "artifacts" / "tokens-v3" / "train_ids.npy"}
OUT = ROOT / "docs" / "measurements" / "chat-template-proof.json"

_Q, _A = "Question: ", "\n\nAnswer: "


def load_tokenizer(tokenizer_dir: Path = TOKENIZER_DIR):
    """The tokenizer exactly as ``train/tokenization.py`` loads it."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)


# ------------------------------------------------------------------------------------
# The two sides of the comparison
# ------------------------------------------------------------------------------------
def pipeline_ids(docs: Sequence[str], tok, *, terminate: bool = True) -> List[int]:
    """Token ids the TRAINING pipeline produces for ``docs``, in order.

    Mirrors ``scripts/prepare_corpus.py`` (normalise, then a ``</s>`` line and a blank line
    after each document) and ``train/tokenization.py`` (the file is read line by line,
    ``rstrip("\\n")``, and each line encoded on its own by ``encode_batch``). With
    ``terminate=False`` the LAST document gets no separator -- an open prompt.
    """
    parts = []
    for i, doc in enumerate(docs):
        parts.append(normalise(doc))
        if terminate or i < len(docs) - 1:
            parts.append("\n" + DOCUMENT_SEPARATOR + "\n\n")
        else:
            parts.append("\n")
    text = "".join(parts)
    lines = [line.rstrip("\n") for line in text.splitlines(keepends=True)]
    return encode_batch(lines, tok).tolist()


def render_ids(messages: List[Dict[str, str]], chat_format: str, tok) -> Tuple[str, List[int]]:
    """(rendered prompt, ids) exactly as a vLLM chat request would produce them."""
    rendered = tok.apply_chat_template(
        messages, chat_template=CHAT_TEMPLATES[chat_format], tokenize=False,
        add_generation_prompt=True,
    )
    return rendered, tok(rendered, add_special_tokens=False)["input_ids"]


# ------------------------------------------------------------------------------------
# Corpus readers
# ------------------------------------------------------------------------------------
def read_documents(path: Path, limit: Optional[int] = None) -> List[str]:
    """Documents of a prepared corpus file (split on the ``</s>`` separator line)."""
    docs: List[str] = []
    buf: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.rstrip("\n") == DOCUMENT_SEPARATOR:
                doc = "".join(buf).strip("\n")
                if doc:
                    docs.append(doc)
                    if limit and len(docs) >= limit:
                        break
                buf = []
            else:
                buf.append(line)
    return docs


def split_dolly(doc: str) -> Optional[Tuple[str, str]]:
    """(user content, assistant content) for a ``_render_dolly`` document, or None.

    The user content is everything between ``Question: `` and the first blank-line
    ``Answer: `` -- the instruction and, when present, the context paragraph, with its
    blank line kept. That is how a client supplies reference text.
    """
    if not doc.startswith(_Q) or _A not in doc:
        return None
    idx = doc.index(_A)
    return doc[len(_Q):idx], doc[idx + len(_A):]


def open_dolly(user: str) -> str:
    """The open-prompt document text for a question: ``Question: ...\\n\\nAnswer:``."""
    return f"{_Q}{user}\n\nAnswer:"


# ------------------------------------------------------------------------------------
# Checks
# ------------------------------------------------------------------------------------
def first_diff(a: Sequence[int], b: Sequence[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def check_dolly(tok, docs: List[str]) -> Dict:
    """Consecutive-pair pipeline diff for ``dolly_qa`` (and the legacy template's distance)."""
    pairs = [split_dolly(d) for d in docs]
    n = matched = prefix_ok = legacy_matched = 0
    mismatches = []
    for i in range(len(docs) - 1):
        a, b = pairs[i], pairs[i + 1]
        if a is None or b is None:
            continue
        n += 1
        msgs = [{"role": "user", "content": a[0]}, {"role": "assistant", "content": a[1]},
                {"role": "user", "content": b[0]}]
        rendered, got = render_ids(msgs, "dolly_qa", tok)
        want = pipeline_ids([docs[i]], tok) + pipeline_ids([open_dolly(b[0])], tok, terminate=False)
        # The open prompt must also be a strict prefix of the full next document's ids --
        # otherwise "what the model continues from" is not what it was trained to continue.
        full_next = pipeline_ids([docs[i + 1]], tok)
        open_only = pipeline_ids([open_dolly(b[0])], tok, terminate=False)
        prefix_ok += full_next[: len(open_only)] == open_only
        if got == want:
            matched += 1
        elif len(mismatches) < 10:
            k = first_diff(got, want)
            mismatches.append({
                "doc_index": i, "first_diff_token": k,
                "rendered_tokens": tok.convert_ids_to_tokens(got[max(0, k - 3): k + 5]),
                "pipeline_tokens": tok.convert_ids_to_tokens(want[max(0, k - 3): k + 5]),
            })
        _, legacy = render_ids(msgs, "tool_call_sft", tok)
        legacy_matched += legacy == want
    return {"pairs_checked": n, "identical": matched, "open_prompt_is_prefix_of_doc": prefix_ok,
            "legacy_tool_call_sft_identical": legacy_matched, "first_mismatches": mismatches}


def check_plain(tok, docs: List[str]) -> Dict:
    """Split each story into an opening line (user) and the rest (assistant)."""
    n = matched = 0
    mismatches = []
    for i, doc in enumerate(docs):
        lines = doc.split("\n")
        if len(lines) < 2:
            continue
        n += 1
        msgs = [{"role": "user", "content": lines[0]},
                {"role": "assistant", "content": "\n".join(lines[1:])}]
        _, got = render_ids(msgs, "plain", tok)
        want = pipeline_ids([doc], tok, terminate=False)
        if got == want:
            matched += 1
        elif len(mismatches) < 10:
            k = first_diff(got, want)
            mismatches.append({"doc_index": i, "first_diff_token": k,
                               "rendered_tokens": tok.convert_ids_to_tokens(got[max(0, k - 3): k + 5]),
                               "pipeline_tokens": tok.convert_ids_to_tokens(want[max(0, k - 3): k + 5])})
    return {"docs_checked": n, "identical": matched, "first_mismatches": mismatches}


def find_in_stream(ids: Sequence[int], stream_path: Path) -> Optional[int]:
    """First offset at which ``ids`` occurs verbatim in a training token stream, or None."""
    a = np.load(stream_path, mmap_mode="r")
    a = np.asarray(a)
    ids = np.asarray(ids, dtype=a.dtype)
    cand = np.flatnonzero(a[: len(a) - len(ids) + 1] == ids[0])
    for k in range(1, min(len(ids), 64)):
        cand = cand[a[cand + k] == ids[k]]
        if cand.size == 0:
            return None
    for c in cand:
        if np.array_equal(a[c: c + len(ids)], ids):
            return int(c)
    return None


def stream_check(tok, chat_format: str, messages: List[Dict[str, str]]) -> Dict:
    rendered, ids = render_ids(messages, chat_format, tok)
    path = STREAMS[chat_format]
    off = find_in_stream(ids, path) if path.is_file() else None
    _, legacy_ids = render_ids(messages, "tool_call_sft", tok)
    legacy_off = find_in_stream(legacy_ids, path) if path.is_file() else None
    return {"stream": str(path.relative_to(ROOT)), "stream_present": path.is_file(),
            "rendered": rendered, "n_tokens": len(ids), "found_at_offset": off,
            "legacy_tool_call_sft_found_at_offset": legacy_off}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=None, help="documents per corpus (default: all)")
    ap.add_argument("--no-stream", action="store_true", help="skip the train_ids.npy search")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    tok = load_tokenizer()
    result: Dict = {"tokenizer": str(TOKENIZER_DIR.relative_to(ROOT)),
                    "how_vllm_tokenizes_chat": "apply_chat_template(tokenize=False) then "
                                               "tokenizer(prompt, add_special_tokens=False)"}

    dolly_docs = read_documents(DIALOGUE, args.limit)
    result["dolly_qa"] = check_dolly(tok, dolly_docs)
    print("dolly_qa:", {k: v for k, v in result["dolly_qa"].items() if k != "first_mismatches"})

    story_docs = read_documents(TINYSTORIES, args.limit or 20000)
    result["plain"] = check_plain(tok, story_docs)
    print("plain:", {k: v for k, v in result["plain"].items() if k != "first_mismatches"})

    if not args.no_stream:
        # Three consecutive dolly docs as a 2-exchange conversation + an open third turn.
        convo = []
        for d in dolly_docs[:2]:
            q, a = split_dolly(d)
            convo += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        convo.append({"role": "user", "content": split_dolly(dolly_docs[2])[0]})
        result["dolly_qa_stream"] = stream_check(tok, "dolly_qa", convo)
        print("dolly_qa stream:", result["dolly_qa_stream"]["found_at_offset"],
              "legacy:", result["dolly_qa_stream"]["legacy_tool_call_sft_found_at_offset"])
        lines = story_docs[0].split("\n")
        story = [{"role": "user", "content": lines[0]},
                 {"role": "assistant", "content": "\n".join(lines[1:])}]
        result["plain_stream"] = stream_check(tok, "plain", story)
        print("plain stream:", result["plain_stream"]["found_at_offset"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
