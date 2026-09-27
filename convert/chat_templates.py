# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The chat templates this project ships, one per training format actually used.

A chat template is not decoration. It is the only thing standing between what an OpenAI
client sends (a list of ``{"role", "content"}`` messages) and the token sequence the model
was trained to continue. If the rendered prompt is a shape the model never saw, the model
does not error -- it quietly does worse. This module exists so that each published model
ships the template that reproduces ITS training data, and so that claim is checkable
(``tests/test_chat_templates.py`` diffs rendered prompts against the training pipeline's
own token ids, and against the real ``train_ids.npy`` streams when they are on disk).

HOW THE TRAINING PIPELINE ACTUALLY TOKENIZES TEXT (the facts every template is built on)
----------------------------------------------------------------------------------------
1. ``scripts/prepare_corpus.py`` writes each document, then a literal ``</s>`` line (the eos
   token, id 2) as the document separator.
2. ``train/tokenization.py`` encodes the corpus **one line at a time**
   (``line.rstrip("\\n")``, ``add_special_tokens=False``). Newlines are therefore never
   tokens: measured over both training streams, ``artifacts/tokens-v3/train_ids.npy`` and
   ``artifacts/tokens-v4/train_ids.npy`` contain **zero** of the 8 vocabulary tokens that
   hold a newline byte. Blank lines encode to nothing.
3. The tokenizer's ByteLevel pre-tokenizer has ``add_prefix_space=True``, so every line
   starts with a leading-space token (``" Answer"`` -> ``ĠAns wer``). ``</s>`` is an added
   token, split out before pre-tokenization, and the text after it gets its own prefix
   space too (verified: ``"a</s>b"`` -> ``Ġa </s> Ġb``).
4. There is no BOS in training (the post-processor is a plain ByteLevel, and the corpus
   carries no ``<s>``): tokens-v4 holds one id-1 token in 352.6M.

Consequence: the token sequence for "line A, line B" in training is exactly the tokenizer's
encoding of the single string ``"A B"`` -- one space between lines, no newline. Every
template below therefore **flattens** message content: split on newlines, trim each line,
drop blank lines, join with one space. A template that emitted ``"\\n"`` would put a
newline token in front of the model, which it has never seen once in 705M training tokens,
and would also cost the following word its prefix space (``"x\\nAnswer:"`` tokenizes as
``Ġx Ċ An s wer :`` -- not ``ĠAns wer :``).

THE FORMATS
-----------
``dolly_qa`` -- tt-tnt-1024 (the designated Stage B checkpoint, trained on
``artifacts/tokens-v4``). The only question-answering data in its corpora is the
databricks-dolly-15k slice, rendered by ``scripts/fetch_corpus.py::_render_dolly`` as::

    Question: {instruction}

    {context}            <- optional reference text

    Answer: {response}
    </s>

i.e. in token space ``" Question: {instruction} [{context}] Answer: {response}" + [2]``.
tokens-v4 contains 38,442 ``" Question:"`` and 38,447 ``" Answer:"`` trigrams, and those
documents sit back to back in the stream (``blend_corpus.py`` writes each source
contiguously), so "a finished Q/A document, then ``</s>``, then the next question" is itself
a sequence the model trained on thousands of times. That is how this template renders a
multi-turn conversation: every completed exchange is one closed dolly document, and the
final user turn is an open one ending in ``" Answer:"`` for the model to continue. The model
then stops by emitting ``</s>`` -- the eos id vLLM already stops on.

Reference text a client wants to supply goes in the same user message after a blank line;
because lines are flattened, it lands exactly where dolly's ``context`` field did.

``plain`` -- tt-tnt (the 22M ``tt-tnt-v3`` checkpoint, trained on ``artifacts/tokens-v3``).
tokens-v3 predates the dolly slice: it holds 10 ``" Question:"`` trigrams in 352.7M tokens
(incidental prose), and no instruction or dialogue data of any kind. There is no
conversational format this model ever saw, so the only honest template is **none**: the
messages are rendered as one continuous piece of plain text for the model to continue (the
way every one of its training documents looks). A user message is a story opening; the
reply is the model's continuation of it. Inventing ``User:``/``Assistant:`` markers here
would be exactly the mistake this module exists to prevent.

``tool_call_sft`` -- the tool-calling SFT checkpoints only (``scripts/eval_improv.py``'s SFT
conversion path). ``train.tool_calling.build_training_text`` renders
``Q: {question}\\nAnswer:{tool_call}`` and ``scripts/train_tool_calling.py`` encodes it as ONE
string (newline token included), so for those checkpoints ``Q: ...\\nAnswer:`` really is
the trained format -- measured: 100% well-formed tool calls through it, 0% through a
``user:``/``assistant:`` template. This is the template tt-tnt-1024 shipped with until
2026-09-27; it was right for the tool-calling checkpoint and wrong for every pretraining
checkpoint since, which never saw ``Q:`` + newline + unspaced ``Answer``.

SHARED RULES
------------
* ``system`` messages are dropped. Neither pretraining corpus has any slot for instructions
  to the model; rendering one would feed it text in a position it has no prior for.
* History is windowed to the last ``MAX_CHAT_TEMPLATE_MESSAGES`` non-system messages -- the
  crash backstop documented on that constant, unchanged.
* ``content`` may be ``None`` (an assistant message carrying only tool calls); it renders as
  empty rather than raising inside the server.
"""
from __future__ import annotations

from typing import Dict, Optional

#: Conservative cap on how many trailing chat messages a shipped chat template renders,
#: regardless of how much history a client sends. Motivated by a real, reproduced serving
#: defect (docs/upstream-tt-metal-asks.md entry 6): a generic tt-metal/vLLM KV-cache bug --
#: confirmed on stock meta-llama/Llama-3.2-1B-Instruct too, so it is not specific to this
#: project's model -- crashes the whole engine on a growing multi-turn conversation well
#: before the rendered prompt approaches the model's own declared context. Reproduced
#: directly: 5 trailing messages (2 completed exchanges + 1 new turn, ~106 tokens on a
#: 512-token model) served successfully; 7 messages crashed. This constant is deliberately
#: set BELOW that observed failure point, not merely "some finite number" -- raising it
#: without re-verifying against the current serving stack would silently reopen the crash.
MAX_CHAT_TEMPLATE_MESSAGES = 5

# The Jinja below is written with ``{%-``/``-%}`` everywhere so that NO whitespace from the
# template source leaks into the rendered prompt: every byte of output is spelled out as a
# literal. That matters because a stray space or newline is a different token sequence.

#: Shared prologue.
#:
#: ``flat(c)`` reproduces what line-by-line training encoding does to a multi-line text:
#: each line loses trailing spaces/tabs -- ONLY those, exactly like
#: ``prepare_corpus.normalise``'s ``[ \t]+$`` (a bare ``rstrip()`` would also eat U+00A0 /
#: U+202F, which normalise keeps; the full-corpus proof caught exactly that on 59 dolly
#: documents) -- plus the ``\r`` of a CRLF line ending, which normalise turns into ``\n``;
#: blank lines vanish (they encode to nothing), and consecutive lines are joined with ONE
#: space -- because each training line was encoded on its own and the tokenizer gave it a
#: prefix space. The one exception is a line that already starts with a space: the
#: tokenizer adds no prefix space to such a line (``add_prefix_space`` only fires when the
#: text does not already start with ``' '``), so it is appended as-is. (A TAB-indented line
#: still gets the prefix space in training, and gets the joining space here.)
#:
#: ``join(a, b)`` is the same rule for gluing two already-flattened pieces.
#:
#: Then: drop system messages and window the rest to ``MAX_CHAT_TEMPLATE_MESSAGES``.
_PROLOGUE = (
    "{%- macro flat(c) -%}"
    "{%- set f = namespace(o='') -%}"
    "{%- for line in (c or '').split('\n') -%}"
    "{%- set l = line.rstrip(' \t\r') -%}"
    "{%- if l -%}"
    "{%- if f.o and not l.startswith(' ') -%}{%- set f.o = f.o + ' ' + l -%}"
    "{%- else -%}{%- set f.o = f.o + l -%}{%- endif -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{{- f.o -}}"
    "{%- endmacro -%}"
    "{%- macro join(a, b) -%}"
    "{{- a + b if (not a or not b or b.startswith(' ')) else a + ' ' + b -}}"
    "{%- endmacro -%}"
    "{%- set msgs = (messages | rejectattr('role', 'equalto', 'system') | list)"
    "[-" + str(MAX_CHAT_TEMPLATE_MESSAGES) + ":] -%}"
)

#: ``dolly_qa``: closed exchanges become ``Question: q Answer: a</s>``; the trailing user
#: turn becomes ``Question: q Answer:``. Both ``q`` and ``a`` are fully stripped, because
#: ``_render_dolly`` strips ``instruction`` and ``response`` (so a reply vLLM returned as
#: ``" Paris"`` re-renders as ``Answer: Paris`` -- exactly the tokens it was generated as).
#: Consecutive user messages are merged into one question with the join rule (as two lines
#: of one dolly instruction would be). An assistant message with no open question before
#: it -- the window cut the question off, or two replies in a row -- is skipped rather than
#: rendered as an answer to nothing.
DOLLY_QA_TEMPLATE = (
    _PROLOGUE
    + "{%- set ns = namespace(q='') -%}"
    "{%- for m in msgs -%}"
    "{%- set text = flat(m['content']) | trim -%}"
    "{%- if m['role'] == 'user' -%}"
    "{%- if text -%}{%- set ns.q = join(ns.q, text) -%}{%- endif -%}"
    "{%- elif m['role'] == 'assistant' and ns.q -%}"
    "Question: {{ ns.q }} Answer:{{ (' ' + text) if text else '' }}</s>"
    "{%- set ns.q = '' -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if ns.q -%}Question: {{ ns.q }} Answer:{%- endif -%}"
)

#: ``plain``: every kept message's flattened text, glued with the join rule, nothing else.
#: Not stripped: a continuation the model produced as ``" there was"`` must re-attach as
#: exactly those tokens.
PLAIN_TEMPLATE = (
    _PROLOGUE
    + "{%- set ns = namespace(out='') -%}"
    "{%- for m in msgs -%}"
    "{%- set ns.out = join(ns.out, flat(m['content'])) -%}"
    "{%- endfor -%}"
    "{{ ns.out }}"
)

#: ``tool_call_sft``: the pre-2026-09-27 template, kept verbatim for the SFT tool-calling
#: checkpoints it was measured on (see the module docstring). Do not use it for a
#: pretraining checkpoint.
TOOL_CALL_SFT_TEMPLATE = (
    "{% set messages = messages[-" + str(MAX_CHAT_TEMPLATE_MESSAGES) + ":] %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}Q: {{ message['content'] }}\nAnswer:"
    "{% else %} {{ message['content'] }}\n{% endif %}"
    "{% endfor %}"
)

#: Format name -> template. The names are what ``publish_to_hub.TARGETS`` and
#: ``convert_checkpoint(chat_format=...)`` refer to.
CHAT_TEMPLATES: Dict[str, str] = {
    "dolly_qa": DOLLY_QA_TEMPLATE,
    "plain": PLAIN_TEMPLATE,
    "tool_call_sft": TOOL_CALL_SFT_TEMPLATE,
}

#: Pre-tokenized corpora whose chat format has been MEASURED (trigram counts over the real
#: ``train_ids.npy``; see the module docstring). A corpus not listed here has no known chat
#: format and gets no template unless the caller names one -- a missing template is an
#: honest HTTP 400 from the server; a guessed one is a silent quality loss.
CORPUS_CHAT_FORMATS: Dict[str, str] = {
    "tokens-v3": "plain",      # 9-source blend, no dialogue slice (tt-tnt-v3 trained here)
    "tokens-v4": "dolly_qa",   # 10-source blend incl. dolly-15k (tt-tnt-1024 Stage B)
}

#: ``corpus_tokens`` recorded in a format-1 checkpoint header (which predates the
#: ``tokens_dir`` field) -> corpus directory. Exact totals, so a match is not a guess:
#: ``artifacts/checkpoints-tt-tnt-v3``'s header records 391,921,555 = tokens-v3's total
#: (and tokens-v4's is 391,823,393 -- the two differ, CLAUDE.md 2026-08-29).
_CORPUS_TOKEN_TOTALS: Dict[int, str] = {
    391_921_555: "tokens-v3",
    391_823_393: "tokens-v4",
}


def template_for(chat_format: str) -> str:
    """The template for a named format. Unknown names raise rather than guess."""
    try:
        return CHAT_TEMPLATES[chat_format]
    except KeyError:
        raise ValueError(
            f"unknown chat_format {chat_format!r}; known: {sorted(CHAT_TEMPLATES)}"
        ) from None


def infer_chat_format(header: Dict) -> Optional[str]:
    """The chat format a checkpoint was trained on, from its header, or ``None``.

    Uses the header's ``tokens_dir`` (format 2) or, failing that, its exact
    ``corpus_tokens`` total (format 1). Only corpora in :data:`CORPUS_CHAT_FORMATS` resolve;
    anything else returns ``None`` so the caller must decide explicitly.
    """
    tokens_dir = header.get("tokens_dir")
    name = str(tokens_dir).rstrip("/").split("/")[-1] if tokens_dir else None
    if name is None:
        try:
            name = _CORPUS_TOKEN_TOTALS.get(int(header.get("corpus_tokens")))
        except (TypeError, ValueError):
            name = None
    return CORPUS_CHAT_FORMATS.get(name) if name else None
