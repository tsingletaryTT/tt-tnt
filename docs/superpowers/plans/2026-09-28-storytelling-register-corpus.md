<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Storytelling Register Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the corpus, tokenizer, and evaluation instrument for the storytelling-register
pretrain — a licence-clean, book-scale narrative source (PG-19) blended with a reweighted
version of the existing curated sources and FineWeb-Edu, a 48K-vocabulary tokenizer, a new
model-size registry entry to match, a small hand-authored "tortoise" flavour slice, and a third
frozen evaluation prompt set built from Tortoise's real discography — everything gates 0–3
require, stopping short of the actual multi-seed training run (a separate follow-up plan).

**Architecture:** Fix a real vocab-size hardcoding bug that would otherwise break any
non-32K-vocab training run; add PG-19 as a new `fetch_kind="url"` corpus source with its own
per-book fetch script (mirroring the existing `mission` source) and a new document-chunking
mechanism (`words_per_document`) so a whole novel doesn't become one document with no interior
`</s>`; add a small hand-authored `tortoise` source the same way; register both in
`train/corpus.py` with reweighted shares; add a new `ModelSize` entry for the 48K vocabulary;
build a third frozen eval prompt set; then run the real pipeline (fetch → measure → settle
shares → blend → retrain tokenizer → tokenize) against real data, producing a corpus and
tokenizer the follow-up training plan trains against.

**Tech Stack:** Python 3.10+, `datasets`, `tokenizers`, `transformers`, `numpy`, `pyyaml`,
`pytest`. No hardware, no new dependencies.

**Spec:** [`docs/superpowers/specs/2026-09-28-storytelling-register-stage-design.md`](../specs/2026-09-28-storytelling-register-stage-design.md)

## Global Constraints

- **SPDX header pair on every new file**; Python 3.10+.
- **No bare `assert` for guards** in production code (tests may assert freely).
- **`pyproject.toml` must NOT be modified**; no new dependencies — everything here uses
  `urllib.request` (stdlib) plus packages already in this project's environment.
- **DELETE NOTHING under `artifacts/`** without the user's explicit instruction. Disk is at
  78 GB free (post-2026-08-31 prune); stop and report rather than reclaim space yourself.
- **Never write under `artifacts/checkpoints*/`, `artifacts/hf-tt-tnt-1024`, or
  `artifacts/hf-tt-tnt-v3`** — protected baseline evidence. Nothing in this plan trains or
  converts a model, so this is a passive guard, not something any task here should ever touch.
- **`target_share` values in `train/corpus.py::SOURCES` must sum to exactly 1.0** at every
  commit (tolerance `1e-9`) — `tests/test_corpus.py::test_target_shares_sum_to_one` enforces
  this and must pass after every task that touches a share.
- **Licensing is DATA, not prose** — lives in `train/corpus.py`'s `CorpusSource` fields,
  rendered into `docs/corpus_licensing.md` by `scripts/render_licensing.py`. Never hand-edit
  that generated file.
- **The corpus is never redistributed**; `artifacts/` stays gitignored. Tracked *records*
  (`docs/measurements/corpus_availability.json`, a new tracked blend manifest) are committed
  copies, per this project's established pattern.
- **`artifacts/tokenizer` is a single shared location, overwritten in place on retrain.**
  Every published HF export directory (`artifacts/hf-tt-tnt-1024`, `artifacts/hf-tt-tnt-v3`)
  carries its own copied-in tokenizer files and is unaffected by retraining this shared
  location — verified precedent, not a new assumption.
- **A `fetch_kind="url"` source needs a resolvable anchor `source_url`**, even when — as for
  `mission`, `pulp_sf`, and both new sources here — the real content comes from a dedicated
  per-source fetch script rather than the generic single-URL path in `scripts/fetch_corpus.py`.
- **Nothing in this plan touches Tenstorrent hardware.** No device open, no `gozer` lease
  needed for any task here — this is corpus/tokenizer/registry work only. The follow-up
  training-run plan is where hardware leasing applies.

## Review Focus

1. **A PG-19 book that is empty or whitespace-only after stripping** must contribute zero
   documents, not a phantom `</s>`-only entry — PG-19's own preparation already strips
   Gutenberg boilerplate per its dataset card, so double-stripping down to nothing is an edge
   case worth handling explicitly, not assuming away.
2. **A `words_per_document` split whose final chunk is very short** (a few words past the last
   whole-paragraph boundary) must still get its own closing separator, matching the existing
   truncated-tail precedent in `blend_corpus.py`, not get silently merged into the prior chunk
   or dropped.
3. **`target_share` drifting away from exactly 1.0** when many small shares are reweighted in
   one edit — the existing test catches this, but it must actually be *run* before committing
   Task 7, not eyeballed.
4. **Anything still comparing a model's real vocabulary against a hardcoded `32000`** after
   Task 1's fix — a repo-wide grep during planning found only the two call sites this plan
   fixes, but that was a point-in-time check, not a guarantee; Task 1's own tests must prove
   the fix by testing a non-32000 vocabulary explicitly, not just that the old default still
   works.
5. **`scripts/build_tokenizer.py --vocab-size 48000` silently under-shooting** (its own
   documented "a ceiling, not a promise" risk) if ever run against a small or partial corpus —
   Task 10 must run it against the real, final ~2.5B-token blend and confirm the existing
   hard-fail path (`achieved_size != args.vocab_size`) is what actually gates it, not a
   convenient early exit on a smaller test corpus.

---

## File Structure

| File | Responsibility |
|---|---|
| `train/config.py` | Deletes the now-dead `VOCAB_SIZE` global (Task 1). |
| `train/checkpoint.py` | `build_header` takes `vocab_size` explicitly (Task 1). |
| `train/run.py` | Vocab-size validation checks the model's own declared size, not a global (Task 1); `build_header` call site passes it explicitly (Task 1). |
| `train/corpus.py` | `CorpusSource.words_per_document` field (Task 2); `pg19` and `tortoise` registered, all shares reweighted (Task 7); real settled shares (Task 9). |
| `scripts/prepare_corpus.py` | Book-chunking (`_split_into_chunks`, `flush_group` change) (Task 2). |
| `scripts/fetch_pg19.py` | New. PG-19 per-book fetch, bypassing `datasets.load_dataset` (Task 3). |
| `train/sizes.py`, `train/configs/model/tt-tnt-1024v48k.yaml` | New 48K-vocabulary model size (Task 4). |
| `scripts/fetch_tortoise.py` | New. Hand-authored vignette corpus, no network (Task 5). |
| `docs/evaluation_prompts_c.json`, `tests/test_evaluation_prompts_c.py` | New. Third frozen prompt set (Task 6). |
| `docs/measurements/corpus_availability.json`, `docs/measurements/blend_manifest.json` | Real, tracked measurement records (Tasks 8, 10). |
| `docs/corpus_licensing.md`, `README.md` | Regenerated / updated provenance (Task 9). |
| `artifacts/tokenizer`, `artifacts/corpus/blend.txt` | Overwritten in place with the new blend and 48K tokenizer (Task 10). |
| `artifacts/tokens-storyreg/{train,val}_ids.npy` | New tokenized corpus (Task 11). |

---

## Task 1: Fix the `vocab_size` hardcoding bug

**Files:**
- Modify: `train/config.py` (delete `VOCAB_SIZE`)
- Modify: `train/checkpoint.py` (`build_header` signature + body + docstring)
- Modify: `train/run.py` (vocab-size validation + `build_header` call site)
- Test: `tests/test_checkpoint.py`, `tests/test_run_validation.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_header(..., vocab_size: int, ...)` — every future caller (including the
  follow-up training-run plan) must pass this explicitly; there is no default.

Every checkpoint header currently records `"vocab_size": VOCAB_SIZE` from a hardcoded global
in `train/config.py` (`= 32000`), regardless of what the model actually was. `train/run.py`
compounds this: it reads the model's *real* declared vocabulary from the model config YAML
into `model_vocab_size`, then raises unless that value equals the same hardcoded global — a
check that is backwards. It should check the token stream against the model's *own* declared
vocabulary, not against an unrelated constant. Training any model whose vocabulary isn't
exactly 32000 (this plan's 48K tokenizer, later) would raise at this line before a device is
even opened.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_checkpoint.py -- replace test_header_records_vocab_and_seq_len_from_config
# with two tests: one for seq_len alone (unchanged behaviour), one proving vocab_size is
# now explicit rather than a global default.

def test_header_records_seq_len_from_config_default():
    from train.config import SEQ_LEN

    h = _header()
    assert h["seq_len"] == SEQ_LEN


def test_header_records_explicit_vocab_size_not_a_global_default():
    """vocab_size must describe the model that produced these weights. A header that
    silently recorded some OTHER model's vocabulary (or a stale global constant) would
    make convert/to_hf.py build a config.json with the wrong embedding-table size, with no
    error anywhere along the way -- the same class of silent lie build_header's seq_len
    handling already guards against."""
    h = _header(vocab_size=48000)
    assert h["vocab_size"] == 48000
```

Also update the `_header()` helper's `base` dict (used by every other test in this file) to
pass `vocab_size=32000` explicitly, since the parameter is now required:

```python
def _header(**kw):
    base = dict(
        step=100,
        model_config_path="/models/nanollama3.yaml",
        tokenizer_dir="artifacts/tokenizer",
        corpus_tokens=127_635_889,
        batch_size=64,
        vocab_size=32000,
    )
    base.update(kw)
    return build_header(**base, seed=0, tokens_dir="artifacts/tokens-test", optimizer={"type": "AdamW"}, ddp=1)
```

And the two other direct `build_header(...)` calls in this file (`_v2_kwargs()`'s dict, if it
exists separately from `_header`, and the bare `build_header(2000, **bad, ...)` call around
line 502) each need `vocab_size=32000` added to whatever kwargs dict they pass — inspect each
call site directly rather than assuming `_header`/`_v2_kwargs` are the only two shapes, since
a missed one fails loudly (a required kwarg with no default) rather than silently.

```python
# tests/test_run_validation.py -- the standalone build_header(...) call around line 334
# needs vocab_size added:

    header = build_header(
        step=1000, model_config_path="m.yaml", tokenizer_dir="tok",
        corpus_tokens=1_000, batch_size=64, seq_len=512, vocab_size=32000,
        extra={"transformer_config": {}, **ttml_cxx_header_fields(SIZES["1024"])}, seed=0, tokens_dir="artifacts/tokens-test", optimizer={"type": "AdamW"}, ddp=1)
    validate_header(header)  # must not raise
    assert header["intermediate_dim"] == 2816
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_checkpoint.py tests/test_run_validation.py -q`
Expected: FAIL — `build_header() missing 1 required keyword-only argument: 'vocab_size'` on
every call site not yet updated, and `test_header_records_explicit_vocab_size_not_a_global_default`
fails with the same error before it even gets to its assertion.

- [ ] **Step 3: Fix `train/checkpoint.py`**

Change the import line, the signature, the body, and the docstring paragraph that currently
justifies reading from the global:

```python
from train.config import SEQ_LEN
```

```python
def build_header(
    step: int,
    *,
    model_config_path: str,
    tokenizer_dir: str,
    corpus_tokens: int,
    batch_size: int,
    seed: int,
    tokens_dir: str,
    optimizer: Dict[str, Any],
    ddp: int,
    vocab_size: int,
    seq_len: int = SEQ_LEN,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the header stored alongside a checkpoint's tensors.

    ``vocab_size`` is now a REQUIRED explicit parameter, not read from a module-level
    global. It used to be read from ``train.config.VOCAB_SIZE``, a hardcoded ``32000`` --
    which recorded the wrong vocabulary for any model whose real embedding table wasn't
    exactly that size, and would have made ``convert/to_hf.py`` build a ``config.json``
    describing a different embedding-table size than the checkpoint's tensors actually
    have, with no error anywhere along the way. The real training call site
    (``train/run.py``) now passes the size actually declared in the model config YAML that
    trained these weights -- the same discipline ``seq_len`` below already follows, for
    exactly the same reason.

    ``seq_len`` records the sequence length **actually used to train this checkpoint**.
    It defaults to ``train.config.SEQ_LEN`` purely for callers (tests, ad-hoc scripts)
    that don't care and don't want to plumb it explicitly — but ``seq_len`` is now a CLI
    flag (``train/run.py --seq-len``), so the module constant is no longer necessarily
    what any given run actually used. The real training call site always passes
    ``seq_len=cfg.seq_len`` explicitly (the resolved ``RunConfig`` value for *this* run),
    precisely so a header never silently records a value the run didn't use.

    ``corpus_tokens`` is the size of the corpus split the checkpoint was trained against
    (train + val token count) — provenance, not a training-volume claim. ``batch_size`` plus
    ``step`` and ``seq_len`` (already in the header) let us record the number that actually
    matters, ``tokens_seen``, without the caller having to compute or pass it separately.

    ``extra`` is also where a caller should put facts that exist only as hardcoded defaults
    in ttml's C++ (e.g. ``intermediate_dim``, ``weight_tying``, ``rms_norm_eps``) and are not
    recoverable from any yaml or from the checkpoint's own tensors later — see
    ``train/run.py``'s call site for why those three specifically must be captured here, at
    write time.
    """
    header: Dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "vocab_size": int(vocab_size),
        "seq_len": int(seq_len),
        "model_config_path": str(model_config_path),
        "tokenizer_dir": str(tokenizer_dir),
        "corpus_tokens": int(corpus_tokens),
        "batch_size": int(batch_size),
        "tokens_seen": int(step) * int(batch_size) * int(seq_len),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "seed": int(seed),
        "tokens_dir": str(tokens_dir),
        "optimizer": dict(optimizer),
        "ddp": int(ddp),
    }
    if extra:
        clashes = sorted(set(extra) & set(_REQUIRED))
        if clashes:
            raise ValueError(f"extra may not override schema field(s): {', '.join(clashes)}")
        header.update(extra)
    return header
```

- [ ] **Step 4: Fix `train/run.py`**

Remove `VOCAB_SIZE` from the `train.config` import list at line 67. Replace the validation
block (currently comparing `model_vocab_size` against the global) with a check against the
model's own declared size, covering both `train_ids` and `val_ids`:

```python
    with model_config.open("r", encoding="utf-8") as f:
        model_yaml = yaml.safe_load(f)
    transformer_config = model_yaml["transformer_config"]
    model_vocab_size = transformer_config["vocab_size"]
    if int(train_ids.max()) >= model_vocab_size:
        raise ValueError(
            f"token id {int(train_ids.max())} in the training split exceeds "
            f"vocab_size {model_vocab_size}; these tokens were produced by a different "
            f"tokenizer than the model config expects"
        )
    if int(val_ids.max()) >= model_vocab_size:
        raise ValueError(
            f"token id {int(val_ids.max())} in the validation split exceeds "
            f"vocab_size {model_vocab_size}; these tokens were produced by a different "
            f"tokenizer than the model config expects"
        )
```

And update the `build_header(...)` call site (around line 1142) to pass the real vocabulary
explicitly:

```python
                header=checkpoint.build_header(
                    step, model_config_path=str(model_config),
                    tokenizer_dir=str(ROOT / "artifacts" / "tokenizer"),
                    corpus_tokens=int(len(train_ids) + len(val_ids)),
                    batch_size=args.batch_size,
                    vocab_size=int(model_vocab_size),
                    seed=int(yaml_config["training_config"]["seed"]),
                    tokens_dir=str(args.tokens_dir),
                    optimizer=yaml_config["training_config"]["optimizer"],
                    ddp=int(args.ddp),
                    seq_len=cfg.seq_len,
                    extra={
                        "transformer_config": transformer_config,
                        **ttml_cxx_header_fields(size),
                    },
                ),
```

- [ ] **Step 5: Delete the now-dead global**

In `train/config.py`, delete the two lines:

```python
#: Must equal the tokenizer's vocabulary (Plan 1 pins it at exactly this).
VOCAB_SIZE = 32000
```

Confirm nothing else imports it: `grep -rn "from train.config import.*VOCAB_SIZE\|train\.config\.VOCAB_SIZE" --include="*.py" .` (excluding `.claude/worktrees/`) must return nothing.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_checkpoint.py tests/test_run_validation.py -q`
Expected: PASS, all tests.

- [ ] **Step 7: Run the full CPU-only suite to catch any other call site**

Run: `python -m pytest -q -x`
Expected: PASS. If anything else imported `train.config.VOCAB_SIZE`, this fails here rather
than at training time.

- [ ] **Step 8: Commit**

```bash
git add train/config.py train/checkpoint.py train/run.py tests/test_checkpoint.py tests/test_run_validation.py
git commit -m "fix: vocab_size in checkpoint headers is explicit, not a hardcoded 32000"
```

---

## Task 2: PG-19 book-chunking support (`words_per_document`)

**Files:**
- Modify: `train/corpus.py` (new `CorpusSource.words_per_document` field + validation)
- Modify: `scripts/prepare_corpus.py` (`_split_into_chunks` + `flush_group` change)
- Test: `tests/test_corpus.py`, `tests/test_prepare_corpus.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `CorpusSource.words_per_document: int = 0`; `prepare_source(name, src, dest,
  rows_per_document=1, words_per_document=0)`; `_split_into_chunks(text: str,
  words_per_document: int) -> List[str]` in `scripts/prepare_corpus.py`. Task 3's PG-19
  registration (Task 7) will set `words_per_document=3000`.

PG-19's documents are whole novels — tens of thousands of words. Treated as one document
each, `</s>` becomes far rarer per token than even the TinyStories-share reduction already
cost this project on termination rate. `rows_per_document` (existing) solves the opposite
problem — grouping several upstream rows that are each *smaller* than a document (poetry's
one-line rows) into one. This task adds the inverse: splitting one upstream row that is
*larger* than a document into several, at paragraph boundaries, each with its own separator.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_corpus.py

def test_words_per_document_defaults_to_disabled():
    from train.corpus import CorpusSource

    src = CorpusSource(name="x", slice="backbone", target_share=0.01,
                       hf_repo="a/b", hf_revision="0" * 40)
    assert src.words_per_document == 0


def test_words_per_document_rejects_negative():
    from train.corpus import CorpusSource

    with pytest.raises(ValueError, match="words_per_document"):
        CorpusSource(name="x", slice="backbone", target_share=0.01,
                     hf_repo="a/b", hf_revision="0" * 40, words_per_document=-1)


def test_words_per_document_cannot_combine_with_grouped_rows():
    """Grouping several rows into one document and then re-splitting that document is
    not a defined operation -- one knob handles rows smaller than a document, the other
    handles rows larger than one, and they are never both true of the same source."""
    from train.corpus import CorpusSource

    with pytest.raises(ValueError, match="rows_per_document"):
        CorpusSource(name="x", slice="backbone", target_share=0.01,
                     hf_repo="a/b", hf_revision="0" * 40,
                     rows_per_document=2, words_per_document=3000)
```

```python
# tests/test_prepare_corpus.py

def test_split_into_chunks_breaks_only_at_paragraph_boundaries():
    from scripts.prepare_corpus import _split_into_chunks

    text = "\n\n".join([
        "one two three four five",      # 5 words
        "six seven eight nine ten",      # 5 words
        "eleven twelve thirteen",        # 3 words
    ])
    chunks = _split_into_chunks(text, words_per_document=8)
    # First chunk closes once it reaches >= 8 words: paragraph 1 (5) is not enough alone,
    # paragraph 2 brings it to 10, which closes the chunk. The short tail (paragraph 3,
    # 3 words) becomes its own chunk rather than being dropped or merged silently.
    assert chunks == [
        "one two three four five\n\nsix seven eight nine ten",
        "eleven twelve thirteen",
    ]


def test_split_into_chunks_handles_a_single_short_document():
    from scripts.prepare_corpus import _split_into_chunks

    assert _split_into_chunks("just one short paragraph", words_per_document=3000) == [
        "just one short paragraph"
    ]


def test_split_into_chunks_on_empty_text_produces_no_chunks():
    """An empty or whitespace-only document (e.g. a PG-19 book that was entirely
    boilerplate) must not produce a phantom separator-only document."""
    from scripts.prepare_corpus import _split_into_chunks

    assert _split_into_chunks("", words_per_document=3000) == []
    assert _split_into_chunks("   \n\n   ", words_per_document=3000) == []


def test_prepare_source_splits_an_oversized_document_into_several(tmp_path):
    src = tmp_path / "raw.jsonl"
    # One row, ~24 words -- split at words_per_document=8 should yield 3 documents.
    long_text = " ".join([f"word{i}" for i in range(24)])
    src.write_text(json.dumps({"text": long_text}) + "\n")
    dest = tmp_path / "out.txt"

    counts = prepare_source("pg19_test", src, dest, rows_per_document=1,
                            words_per_document=8)

    assert counts["documents"] == 3
    body = dest.read_text()
    assert body.count(DOCUMENT_SEPARATOR) == 3


def test_prepare_source_with_words_per_document_zero_is_unchanged(tmp_path):
    """The default path (words_per_document=0) must produce byte-identical output to
    before this task -- one document per row-group, exactly as prepare_source already
    does for every existing source."""
    src = tmp_path / "raw.jsonl"
    src.write_text(json.dumps({"text": "a short document"}) + "\n")
    dest = tmp_path / "out.txt"

    counts = prepare_source("x", src, dest, rows_per_document=1, words_per_document=0)

    assert counts["documents"] == 1
    assert dest.read_text() == f"a short document\n{DOCUMENT_SEPARATOR}\n\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_corpus.py -k words_per_document tests/test_prepare_corpus.py -k "split_into_chunks or splits_an_oversized or with_words_per_document_zero" -q`
Expected: FAIL — `AttributeError`/`TypeError` (field and function don't exist yet).

- [ ] **Step 3: Add the field to `CorpusSource`**

In `train/corpus.py`, add after `rows_per_document`:

```python
    #: Split an oversized upstream document into several ~N-word sub-documents, each
    #: terminated with its own separator, instead of writing it as one document. 0 (the
    #: default) disables splitting.
    #:
    #: The OPPOSITE case from ``rows_per_document``: that field groups several upstream
    #: rows that are each SMALLER than a document (poetry's one-line rows) into one.
    #: This field splits a single upstream row that is LARGER than a document (a whole
    #: novel, e.g. PG-19) into several -- otherwise the row's one closing separator is
    #: the only ``</s>`` in tens of thousands of words, which is the same termination-rate
    #: failure mode this project already measured when TinyStories' share was cut (it
    #: supplied 73% of the blend's document separators; thinning it cost termination
    #: rate). See ``scripts/prepare_corpus.py::_split_into_chunks``.
    words_per_document: int = 0
```

Add validation to `__post_init__`, after the existing `fetch_kind` checks:

```python
        if self.words_per_document < 0:
            raise ValueError(
                f"{self.name}: words_per_document must be >= 0, got "
                f"{self.words_per_document}"
            )
        if self.words_per_document and self.rows_per_document != 1:
            raise ValueError(
                f"{self.name}: words_per_document and rows_per_document > 1 cannot be "
                f"combined -- grouping several rows into one document and then "
                f"re-splitting that document is not a defined operation"
            )
```

- [ ] **Step 4: Add the splitting logic to `scripts/prepare_corpus.py`**

Add `_split_into_chunks` near `normalise`:

```python
def _split_into_chunks(text: str, words_per_document: int) -> List[str]:
    """Split ``text`` into chunks of approximately ``words_per_document`` words each,
    breaking only at paragraph boundaries (blank lines) so no paragraph is cut mid-way.

    Whole paragraphs accumulate into the current chunk until the running word count meets
    or exceeds the target; the paragraph that crosses the threshold closes the chunk
    rather than starting the next one. A short final paragraph or two becomes its own
    (possibly short) closing chunk rather than being dropped or silently merged -- the
    same "close the tail, don't lose it" rule ``scripts/blend_corpus.py``'s truncated-pass
    handling already follows for the blend's own seams.

    Returns an empty list for empty or whitespace-only text, so a document that is
    nothing after Gutenberg boilerplate stripping produces zero phantom documents rather
    than one separator-only entry.
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0
    for para in paragraphs:
        current.append(para)
        current_words += len(para.split())
        if current_words >= words_per_document:
            chunks.append("\n\n".join(current))
            current = []
            current_words = 0
    if current:
        chunks.append("\n\n".join(current))
    return chunks
```

Add the `List` import if not already present (`from typing import NamedTuple` currently —
change to `from typing import List, NamedTuple`).

Modify `prepare_source`'s signature and its `flush_group`:

```python
def prepare_source(name: str, src: Path, dest: Path, rows_per_document: int = 1,
                   words_per_document: int = 0) -> dict:
```

(Add this to the existing docstring: "``words_per_document`` (from the registry) splits an
oversized document into several sub-documents at paragraph boundaries instead of writing it
as one — see ``CorpusSource.words_per_document``. 0 (the default) disables this and
reproduces the exact prior behaviour.")

```python
        def flush_group() -> None:
            """Write the buffered rows as one or more separator-terminated documents.

            Normally the whole group is one document. When ``words_per_document`` is set
            (PG-19: a book is one row/group, tens of thousands of words), the group is
            instead split into several ~words_per_document-word sub-documents, each
            getting its own separator.
            """
            if not group:
                return
            joined = "\n\n".join(group)
            pieces = (_split_into_chunks(joined, words_per_document)
                     if words_per_document > 0 else [joined])
            for piece in pieces:
                fout.write(piece)
                fout.write("\n" + DOCUMENT_SEPARATOR + "\n\n")
                counts["documents"] += 1
            group.clear()
```

Update `main()`'s call site to pass it through:

```python
        counts = prepare_source(name, src, dest,
                                rows_per_document=source.rows_per_document,
                                words_per_document=source.words_per_document)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_corpus.py tests/test_prepare_corpus.py -q`
Expected: PASS, all tests including the pre-existing ones (the default `words_per_document=0`
path must remain byte-identical to before this task).

- [ ] **Step 6: Commit**

```bash
git add train/corpus.py scripts/prepare_corpus.py tests/test_corpus.py tests/test_prepare_corpus.py
git commit -m "feat: split oversized documents at paragraph boundaries (words_per_document)"
```

---

## Task 3: `scripts/fetch_pg19.py`

**Files:**
- Create: `scripts/fetch_pg19.py`
- Test: `tests/test_fetch_pg19.py`

**Interfaces:**
- Consumes: `scripts.fetch_corpus.write_documents` (existing), `train.paths.shared_dir`
  (existing).
- Produces: `PG19_REVISION: str`, `fetch_file_list() -> List[str]`,
  `iter_pg19_rows(limit_books: int = 0, file_list: Optional[List[str]] = None) -> Iterator[Dict]`,
  `main()`. Task 7 references `PG19_REVISION` when registering the source.

PG-19 (`deepmind/pg19`, dataset repo `4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b`, verified via
the HF Hub API) is a `GeneratorBasedBuilder` (script-based) dataset — `datasets.load_dataset`
would need `trust_remote_code=True`, which nothing in this project's fetch pipeline passes.
Reading `pg19.py` directly (not running it) shows it does nothing more than read
`data/train_files.txt` (28,602 relative paths, e.g. `train/10.txt`) and fetch each book's
plain text from `https://storage.googleapis.com/deepmind-gutenberg/<path>`. This script does
that fetch directly — the same shape as `scripts/fetch_mission.py`, at 28,602 pages instead
of one, and each book is already exactly one document (`rows_per_document` stays the default
1; `words_per_document` from Task 2 handles the chunking downstream in `prepare_corpus.py`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fetch_pg19.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""No network here: these tests exercise fetching through fixtures and monkeypatching,
never a live request -- the same discipline tests/test_fetch_mission.py already follows."""
from scripts.fetch_pg19 import PG19_REVISION, _ASSET_ROOT_URL, iter_pg19_rows


def test_revision_is_a_full_git_sha():
    assert len(PG19_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in PG19_REVISION)


def test_asset_root_is_the_bucket_pg19s_own_loading_script_names():
    """Verified by reading pg19.py directly (deepmind/pg19 at PG19_REVISION), not assumed
    from the dataset card's prose."""
    assert _ASSET_ROOT_URL == "https://storage.googleapis.com/deepmind-gutenberg/"


def test_iter_pg19_rows_fetches_each_listed_book_in_order(monkeypatch):
    fetched_urls = []

    def fake_fetch(url, timeout=120):
        fetched_urls.append(url)
        return f"book text for {url}"

    monkeypatch.setattr("scripts.fetch_pg19._fetch_text", fake_fetch)
    rows = list(iter_pg19_rows(file_list=["train/10.txt", "train/11.txt"]))
    assert [r["text"] for r in rows] == [
        "book text for https://storage.googleapis.com/deepmind-gutenberg/train/10.txt",
        "book text for https://storage.googleapis.com/deepmind-gutenberg/train/11.txt",
    ]
    assert fetched_urls == [
        "https://storage.googleapis.com/deepmind-gutenberg/train/10.txt",
        "https://storage.googleapis.com/deepmind-gutenberg/train/11.txt",
    ]


def test_iter_pg19_rows_respects_limit_books(monkeypatch):
    monkeypatch.setattr("scripts.fetch_pg19._fetch_text", lambda url, timeout=120: "x")
    rows = list(iter_pg19_rows(
        limit_books=1, file_list=["train/10.txt", "train/11.txt", "train/12.txt"]))
    assert len(rows) == 1


def test_iter_pg19_rows_skips_a_book_that_fetches_empty(monkeypatch, capsys):
    monkeypatch.setattr("scripts.fetch_pg19._fetch_text", lambda url, timeout=120: "   ")
    rows = list(iter_pg19_rows(file_list=["train/10.txt"]))
    assert rows == []
    assert "WARNING" in capsys.readouterr().err


def test_iter_pg19_rows_defaults_to_fetching_the_real_file_list_when_none_given(monkeypatch):
    """Proves the wiring without a network call: fetch_file_list is what gets invoked
    when file_list is omitted, not silently skipped."""
    called = []
    monkeypatch.setattr("scripts.fetch_pg19.fetch_file_list", lambda: called.append(1) or ["train/1.txt"])
    monkeypatch.setattr("scripts.fetch_pg19._fetch_text", lambda url, timeout=120: "text")
    rows = list(iter_pg19_rows())
    assert called == [1]
    assert len(rows) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fetch_pg19.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fetch_pg19'`.

- [ ] **Step 3: Write `scripts/fetch_pg19.py`**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Fetch PG-19 books into ``artifacts/raw/pg19/text.jsonl``.

PG-19 (``deepmind/pg19``, dataset repo commit ``4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b`` --
verified via the HF Hub API, ``https://huggingface.co/api/datasets/deepmind/pg19`` -- Apache-2.0
packaging over pre-1919 Project Gutenberg public-domain texts, per its dataset card) is a
``GeneratorBasedBuilder`` (script-based) HuggingFace dataset. ``datasets.load_dataset(
"deepmind/pg19", streaming=True)`` would require ``trust_remote_code=True`` in current
``datasets`` releases, which nothing else in this project's fetch pipeline passes, and
executing an upstream dataset's arbitrary Python is a different trust boundary than
downloading a plain-text file.

Reading PG-19's own loading script directly (not running it) shows it does nothing more
than: read ``data/train_files.txt`` from the dataset repo (one relative path per book, e.g.
``train/10.txt``, 28,602 lines for the train split) and fetch each book's plain text from
``https://storage.googleapis.com/deepmind-gutenberg/<path>``. This script does exactly that
fetch, directly -- the same shape ``scripts/fetch_mission.py`` already uses for a single-page
NASA-transcript source, at 28,602 pages instead of one.

The file list is fetched from the dataset repo at the pinned commit above, so which BOOKS
this run can draw from is reproducible; the book BODIES themselves come from a Google Cloud
Storage bucket this project does not control the pinning of, which is the same class of
residual risk already accepted for the ``mission`` source's NASA URL.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.fetch_corpus import write_documents  # noqa: E402
from train.paths import shared_dir  # noqa: E402

#: The dataset repo commit this fetch is pinned to. Re-verify via the HF Hub API before
#: changing it: the file list (not just the book bodies) is read at this exact commit.
PG19_REVISION = "4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b"

_FILE_LIST_URL = (
    f"https://huggingface.co/datasets/deepmind/pg19/raw/{PG19_REVISION}/data/train_files.txt"
)
#: Where PG-19's own loading script (verified by reading ``pg19.py`` directly) says every
#: book body is actually served from.
_ASSET_ROOT_URL = "https://storage.googleapis.com/deepmind-gutenberg/"


def _fetch_text(url: str, timeout: int = 120) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        raw = fh.read()
    return raw.decode("utf-8", errors="replace")


def fetch_file_list(timeout: int = 120) -> List[str]:
    """The pinned list of relative book paths, one per line (e.g. ``train/10.txt``)."""
    text = _fetch_text(_FILE_LIST_URL, timeout=timeout)
    return [line.strip() for line in text.splitlines() if line.strip()]


def iter_pg19_rows(limit_books: int = 0,
                   file_list: Optional[List[str]] = None) -> Iterator[Dict[str, object]]:
    """Fetch each book in ``file_list`` (or the live pinned list) as one ``{"text": ...}`` row.

    ``limit_books`` caps how many books are fetched (0 = all 28,602) -- PG-19 offers far
    more raw text than any planned share of this blend needs, so a real run passes an
    explicit cap. ``file_list`` is accepted as a parameter (rather than always calling
    :func:`fetch_file_list`) so callers and tests can supply a fixed list with no network
    access for the list itself.
    """
    names = file_list if file_list is not None else fetch_file_list()
    if limit_books:
        names = names[:limit_books]
    for relpath in names:
        url = _ASSET_ROOT_URL + relpath
        text = _fetch_text(url)
        if not text.strip():
            print(f"WARNING: {relpath} produced no text", file=sys.stderr)
            continue
        yield {"text": text}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit-books", type=int, default=0,
                   help="Fetch only the first N books (0 = all 28,602). PG-19 offers far "
                        "more text than any planned share needs; cap this for a real run.")
    p.add_argument("--out", type=Path, default=None,
                   help="Override the destination path (default: artifacts/raw/pg19/text.jsonl).")
    args = p.parse_args()

    dest = args.out or (shared_dir("raw") / "pg19" / "text.jsonl")
    n = write_documents(iter_pg19_rows(args.limit_books), dest)
    print(f"pg19: {n:,} documents -> {dest}")
    if n == 0:
        print("WARNING: pg19 produced no documents", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fetch_pg19.py -q`
Expected: PASS, all 5 tests, zero network calls.

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_pg19.py tests/test_fetch_pg19.py
git commit -m "feat: add scripts/fetch_pg19.py, fetching PG-19 books directly by URL"
```

---

## Task 4: New model size for the 48K vocabulary (`1024v48k`)

**Files:**
- Create: `train/configs/model/tt-tnt-1024v48k.yaml`
- Modify: `train/sizes.py` (new `SIZES["1024v48k"]` entry)

**Interfaces:**
- Consumes: `train.sizes.ModelSize` (existing).
- Produces: `SIZES["1024v48k"]`, selectable via `train/run.py --size 1024v48k` in the
  follow-up training-run plan.

`ModelSize.vocab_size` is a field tied 1:1 to a specific YAML — the existing `"1024"` entry
(`vocab_size=32000`) describes every checkpoint this project has ever trained at that shape,
including the published `episod/tt-tnt-1024`. Changing it in place would retroactively
misdescribe all of that history and break `tests/test_sizes.py::test_registry_matches_its_yaml`
for the existing checkpoints' own YAML. A new size, identical in every other dimension, is the
established pattern (`"384s512"` for a sequence-length variant) — this project's own
convention states adding a size means "a new YAML..., a new entry..., and nothing else",
and `tests/test_sizes.py`'s parametrised tests (`ALL_SIZES = sorted(SIZES)`) cover it for free.

- [ ] **Step 1: Write the new YAML**

```yaml
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# tt-tnt-1024v48k — identical shape to tt-tnt-1024, 48,000-token vocabulary.
#
# See docs/superpowers/specs/2026-09-28-storytelling-register-stage-design.md. The larger,
# more lexically diverse narrative blend this size trains against (PG-19 plus the existing
# curated sources plus FineWeb-Edu) is expected to exhaust a 32,000-token vocabulary's
# headroom more than the prior, TinyStories-heavy blend did; 48,000 gives real margin,
# checked against the achieved vocabulary at build time (scripts/build_tokenizer.py already
# hard-fails on a shortfall). Every other dimension is unchanged from tt-tnt-1024: same
# multi-chip-capable shape (num_groups: 4 admits mesh widths {1, 2, 4}), same grid-fit
# rationale for the derived FFN. See train/sizes.py for the full argument, which is not
# repeated here to avoid two divergent copies of the same reasoning.
transformer_config:
  model_type: "llama"
  num_heads: 16
  num_groups: 4
  embedding_dim: 1024
  dropout_prob: 0.0
  num_blocks: 8
  vocab_size: 48000
  max_sequence_length: 512
  runner_type: default
  theta: 500000.0
```

- [ ] **Step 2: Add the registry entry**

In `train/sizes.py`, add to `SIZES` (after `"1024"`):

```python
    "1024v48k": ModelSize(
        name="1024v48k",
        embedding_dim=1024,
        num_blocks=8,
        num_heads=16,
        num_groups=4,
        vocab_size=48000,
        max_sequence_length=512,
        theta=500000.0,
        rationale=(
            "Identical shape to '1024' (same multi-chip-capable mesh widths {1,2,4}, same "
            "grid-fit FFN derivation), 48,000-token vocabulary instead of 32,000 -- see "
            "docs/superpowers/specs/2026-09-28-storytelling-register-stage-design.md. A new "
            "size rather than editing '1024' in place: vocab_size sizes the embedding table, "
            "so every checkpoint '1024' has ever produced (including the published "
            "episod/tt-tnt-1024) would be retroactively misdescribed by changing it there. "
            "max_sequence_length stays 512, matching '1024': this plan does not revisit "
            "context length (see the spec's 'What is already eliminated' section -- two "
            "2048-context retrains were tried and reverted for this line already)."
        ),
    ),
```

- [ ] **Step 3: Run tests to verify the new size is covered**

Run: `python -m pytest tests/test_sizes.py -q`
Expected: PASS. `test_every_registered_size_has_a_config_file`,
`test_registry_matches_its_yaml`, `test_no_stray_config_files`,
`test_dimensions_are_tile_aligned`, and `test_heads_divide_the_hidden_dimension` all
parametrise over `sorted(SIZES)` and now include `"1024v48k"` automatically.

- [ ] **Step 4: Commit**

```bash
git add train/configs/model/tt-tnt-1024v48k.yaml train/sizes.py
git commit -m "feat: add tt-tnt-1024v48k model size (1024 shape, 48K vocabulary)"
```

---

## Task 5: `scripts/fetch_tortoise.py` — the hand-authored vignette corpus

**Files:**
- Create: `scripts/fetch_tortoise.py`
- Test: `tests/test_fetch_tortoise.py`

**Interfaces:**
- Consumes: `scripts.fetch_corpus.write_documents` (existing), `train.paths.shared_dir`
  (existing).
- Produces: `TORTOISE_VIGNETTES: List[Tuple[str, str]]` (title, vignette text),
  `iter_tortoise_rows() -> Iterator[Dict]`, `main()`. Task 7 registers this source in
  `train/corpus.py`.

Thirty short, evocative vignettes, one per verified real Tortoise (the band) instrumental
track title — deliberately not TinyStories-simple, since this slice's whole purpose is a
register counterweight, not more backbone prose. Every title below was verified against the
band's real discography during this plan's brainstorming (cross-checked Wikipedia, Discogs,
AllMusic); two tracks confirmed to carry vocals (*Rock On*, *Yonder Blue*, both off *The
Catastrophist*) are excluded, matching this slice's "instrumental titles" framing. No
network fetch happens here at all — this mirrors `mission`'s and `pulp_sf`'s precedent of a
`fetch_kind="url"` registry entry whose real content lives in a dedicated per-source script,
except this script's "fetch" is simply writing out hand-authored text.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fetch_tortoise.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""No network here: this source is hand-authored text, not fetched from anywhere."""
from scripts.fetch_tortoise import TORTOISE_VIGNETTES, iter_tortoise_rows

#: Confirmed carrying vocals (The Catastrophist, 2016) -- excluded from this
#: instrumental-titles slice. Kept here as a negative-space check, not a note.
_VOCAL_TRACKS = {"Rock On", "Yonder Blue"}


def test_thirty_vignettes_are_registered():
    assert len(TORTOISE_VIGNETTES) == 30


def test_every_title_is_distinct():
    titles = [title for title, _ in TORTOISE_VIGNETTES]
    assert len(titles) == len(set(titles))


def test_no_vocal_track_is_included():
    titles = {title for title, _ in TORTOISE_VIGNETTES}
    assert not (titles & _VOCAL_TRACKS)


def test_every_vignette_has_real_length():
    """A few hundred words each, not a placeholder sentence."""
    for title, text in TORTOISE_VIGNETTES:
        word_count = len(text.split())
        assert word_count >= 120, f"{title!r}: only {word_count} words"


def test_iter_tortoise_rows_yields_one_row_per_vignette():
    rows = list(iter_tortoise_rows())
    assert len(rows) == len(TORTOISE_VIGNETTES)
    assert all(isinstance(r["text"], str) and r["text"].strip() for r in rows)


def test_iter_tortoise_rows_text_matches_the_registered_vignette():
    rows = list(iter_tortoise_rows())
    for (_, vignette), row in zip(TORTOISE_VIGNETTES, rows):
        assert row["text"] == vignette
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fetch_tortoise.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fetch_tortoise'`.

- [ ] **Step 3: Write `scripts/fetch_tortoise.py`**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Write the hand-authored "tortoise" vignette corpus to ``artifacts/raw/tortoise/text.jsonl``.

Thirty short vignettes, each seeded by a real, verified instrumental track title from
Tortoise (the Chicago post-rock band) -- cross-checked against Wikipedia, Discogs, and
AllMusic during this plan's brainstorming, spanning all seven studio albums (Tortoise 1994,
Millions Now Living Will Never Die 1996, TNT 1998, Standards 2001, It's All Around You 2004,
Beacons of Ancestorship 2009, The Catastrophist 2016). Two tracks confirmed to carry vocals
(Rock On, Yonder Blue, both off The Catastrophist) are excluded, matching this slice's
"instrumental titles" framing.

This project's own name is understood, per the spec above, as TT (Tenstorrent) + Tortoise's
album TNT -- and "A Simple Way to Go Faster Than Light That Does Not Work" (TNT, 1998) is the
real origin of the faster-than-light canary prompt this project has run at every checkpoint
since the qualitative-canary convention started.

No network fetch happens here: unlike every other source, this one is hand-authored text
written directly for this project, not drawn from an external corpus. It is registered as
``fetch_kind="url"`` in ``train/corpus.py`` (a resolvable anchor, never itself fetched) the
same way ``mission`` and ``pulp_sf`` are -- the real content lives in this dedicated script,
matching that established precedent.

The vignettes are deliberately NOT TinyStories-simple: this slice's whole purpose is a
register counterweight (evocative, imagistic prose), not more backbone readability, so its
vocabulary and sentence shape lean further than the rest of this corpus's curated sources.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.fetch_corpus import write_documents  # noqa: E402
from train.paths import shared_dir  # noqa: E402

#: (title, vignette) pairs. Titles are real, verified Tortoise instrumental track titles;
#: vignettes are original text written for this project, using each title as a narrative
#: seed rather than quoting or describing the song itself.
TORTOISE_VIGNETTES: List[Tuple[str, str]] = [
    (
        "A Simple Way to Go Faster Than Light That Does Not Work",
        "The instructions were seven pages long and the seventh page was blank, which "
        "the apprentice took as either an ending or a warning. He built the device from "
        "copper wire and a length of pipe that had once carried steam to a laundry that "
        "no longer existed, and on the appointed evening he stood inside the ring of "
        "chalk and switched it on. Nothing rushed past him. Nothing dimmed or brightened. "
        "The candle on the table kept its ordinary flicker, and the moths outside kept "
        "their ordinary circling of the porch light, unhurried, unconcerned with anyone's "
        "seventh page. He stood there for eleven minutes by the kitchen clock, certain "
        "that a correct machine announces itself by silence rather than by thunder, and "
        "that the silence he was hearing was the specific, patient silence of a wrong "
        "wire. He wrote in his notebook: not light, not sound, not even smoke -- only the "
        "small disappointment of a thing behaving exactly as it always had. Then he "
        "unscrewed the pipe, coiled the wire, and put both away in the drawer where the "
        "other simple ways were already resting, each labelled in his own hand with the "
        "date it had failed to work, each failure filed as carefully as a success would "
        "have been, because a method that does not work is still a method, and someone, "
        "someday, opening that drawer, would need to know exactly which roads had already "
        "been walked to their quiet, ordinary ends."
    ),
    (
        "Djed",
        "The pillar had stood in the museum's basement for forty years, uncatalogued, "
        "because no one on staff could agree on what it was a pillar OF. It was carved "
        "in four tiers like a stack of wheels, each tier narrower than the one below, and "
        "the guide who found it while searching for a spare projector bulb ran her hand "
        "along the lowest tier and felt, very faintly, a warmth that had no business "
        "being there in an unheated room. She did not report the warmth. She reported the "
        "pillar. Three curators came down with flashlights and a tape measure and agreed, "
        "after an hour, that it resembled the djed of old temple carvings -- the backbone "
        "of a god, raised upright as a promise that a fallen thing could stand again -- "
        "though none of them could say how a fourth-tier basalt column had arrived in a "
        "coastal city with no record of purchase, donation, or theft. It went on display "
        "the following spring with a small card reading ORIGIN UNKNOWN, and visitors began, "
        "without instruction, to touch the lowest tier on their way past, the way people "
        "touch a lucky stone in a cathedral floor. The warmth never left it. No one who "
        "touched it ever mentioned the warmth to anyone else, as though each visitor "
        "believed, privately, that they alone had felt the pillar answer."
    ),
    (
        "Glass Museum",
        "Every case in the collection held something that used to hold something else: a "
        "bottle that had carried medicine for a fever no longer diagnosed, a pane that had "
        "once been a window in a house since demolished, a paperweight with a storm frozen "
        "at its centre that the donor swore had been a real storm, shrunk down and sealed "
        "in, though the placard beneath it wisely declined to confirm this. The curator "
        "walked the floor each evening after closing, not to inspect the glass but to "
        "listen to it, because on quiet nights the whole room produced a low, continuous "
        "hum, as if every case were still remembering the shape of whatever it used to "
        "contain. She had never told a visitor this. She had barely told herself, preferring "
        "to file it under acoustics, under old pipes, under the ordinary strangeness of an "
        "old building settling into its foundations. But tonight the hum had a shape to it, "
        "almost a rhythm, rising from the west wall where the oldest pieces sat -- Roman "
        "fragments, mostly, too cloudy now to see through -- and she found herself standing "
        "very still in the dark, waiting for the hum to resolve into something she could "
        "name, the way you wait, at the edge of sleep, for a half-heard sentence to finish "
        "itself and tell you, finally, what it had been trying to say all along."
    ),
    (
        "Swung from the Gutters",
        "The boy climbed because the roofline made more sense to him than the street did: "
        "up there, the town's crooked geometry resolved into straight tin ridges and true "
        "right angles, and a person could travel from the bakery to the schoolhouse without "
        "once touching the mud that swallowed boots to the ankle after rain. He went hand "
        "over hand along the gutters, which groaned under his weight but held, rust flaking "
        "onto his palms like a second, borrowed skin. From up there he could see which "
        "chimneys smoked and which didn't, which shutters had been painted twice and left "
        "unfinished, which yards kept dogs and which kept only the memory of a dog, a "
        "chain still bolted to a post with nothing on the end of it. He told no one about "
        "the route. It was not a secret so much as a language only his hands and feet knew "
        "how to speak, one gutter-length at a time, and when at last he reached the far "
        "roof above the schoolhouse and lowered himself down through its skylight into the "
        "empty classroom, dust motes turning gold in the last of the daylight, he understood "
        "that he had arrived not early and not late but simply by a different clock, the "
        "one that measured a town by its rooftops instead of its roads."
    ),
    (
        "Ten-Day Interval",
        "Every ten days, without fail, a letter arrived at the lighthouse addressed to no "
        "one in particular, and every ten days the keeper opened it anyway, because someone "
        "had to, and because the envelopes were never sealed, only folded, as if inviting "
        "exactly that. The letters described weather that had not yet happened -- a fog "
        "bank on a Tuesday, a hard frost on the following Friday -- and for eleven years the "
        "keeper had watched the described weather arrive precisely ten days later, not a day "
        "early, not a day late, until the interval itself felt less like a delay than like a "
        "kind of breathing, the sea inhaling a forecast and exhaling, ten days on, the fact "
        "of it. He never learned who wrote the letters or how they reached the rock without "
        "a boat, a gull, or a name on the postmark. He stopped asking around year four. What "
        "remained, by year eleven, was a private arithmetic: today's fog was ten days old "
        "news, and somewhere out past the horizon, today's calm was already being folded "
        "into an envelope for a keeper he would never meet, in a life the interval had not "
        "yet delivered."
    ),
    (
        "I Set My Face to the Hillside",
        "She turned away from the harbour lights on purpose, because the harbour was where "
        "everyone else was looking, and she had come to believe that the thing worth finding "
        "was always behind you, in the direction nobody thought to check twice. The hillside "
        "climbed in uneven terraces, old field walls gone soft with moss, and she set her "
        "face to it the way a sailor sets a course, deliberately, as an instrument is set, "
        "and did not look back at the boats or the lanterns or the noise of the evening "
        "market breaking up below. Halfway up she found a door standing alone with no house "
        "attached to it, weathered grey, hinges intact, opening onto nothing but more hill. "
        "She opened it anyway. On the far side the grass was the same grass, the sky the "
        "same sky, and yet something about passing through the frame made the climb feel "
        "sanctioned, as though the hill had been waiting for someone to use its one correct "
        "entrance instead of simply walking around. She kept climbing past the door until "
        "the harbour was a handful of gold specks and the wind carried no voices at all, "
        "only the interval between one gust and the next, and she understood that this, "
        "not the summit, was the place she had actually set out to find."
    ),
    (
        "The Equator",
        "The old surveyor kept a brass rod exactly one metre long and swore he could feel "
        "the line itself underfoot when he crossed it, a faint change in the pull of things, "
        "though every instrument he owned insisted the ground was ground the same as any "
        "other ground. He had crossed it eleven times in his career, always on foot, never "
        "by the marked road with its concrete monument and its photograph stand, because he "
        "did not trust a line drawn by a government to be the same line drawn by the planet. "
        "His own method was simpler: he walked north until the rod, balanced on one finger "
        "at noon, cast no shadow he could not explain, and there, in a field or a ditch or "
        "once in the middle of someone's cassava patch, he would drive a small stake and "
        "call it correct. The farmers he startled generally forgave him. One gave him lunch. "
        "He kept a ledger of every crossing, eleven entries, eleven slightly different "
        "latitudes, and told anyone who asked that the line was not a place but a "
        "negotiation, renewed every time a man with a brass rod and too much patience went "
        "looking for the exact middle of everything and found, instead, eleven approximate "
        "answers he had learned to love equally."
    ),
    (
        "The Suspension Bridge at Iguazú Falls",
        "The bridge had been built to survive the spray, which over a century had rusted "
        "three previous crossings into unusable lace, and the engineers had learned, "
        "finally, to let the cables sing instead of fighting the wind that came off the "
        "falls in gusts strong enough to lean a grown man sideways. Tourists crossed it "
        "gripping the rails with both hands, faces wet before they'd gone ten steps, unable "
        "to hear each other over the roar rising from below, where the river simply ended "
        "and became, for a moment, entirely vertical. A guide walked the bridge every "
        "morning before the gates opened, alone, and had come to know its particular "
        "vocabulary of creaks -- which one meant a cable adjusting to the night's cooling, "
        "which one meant nothing at all, which one, twice in eleven years, had meant a bolt "
        "worth reporting. She liked the bridge best in that empty hour, spray drifting up "
        "instead of falling down, rainbows forming and dissolving in the mist with no one "
        "there to photograph them, as if the falls kept a private version of themselves for "
        "whoever arrived first and left it, generously, unphotographed and complete, before "
        "the gates opened and the bridge began to carry, all day, the weight of everyone "
        "else's astonishment."
    ),
    (
        "Four-Day Interval",
        "The garden kept its own schedule, four days between one strange bloom and the "
        "next, and the old woman who tended it had stopped fighting the schedule sometime "
        "around her sixtieth birthday. On the first day, nothing. On the second, a bud "
        "swelling somewhere she was certain hadn't held a bud the evening before. On the "
        "third, colour, always a colour the seed packet had never promised. On the fourth, "
        "the flower opened fully, held its shape for exactly one day, and by the fifth "
        "morning had folded itself back down into a green, unremarkable bud again, as "
        "though it had never bloomed at all, waiting out its four days before trying once "
        "more. She had given up naming the flowers years ago; naming implied a single "
        "flower doing a single thing, and this was closer to a conversation the garden was "
        "having with itself, four beats to a sentence, and her job, she had decided, was "
        "simply to be present on the fourth day, to witness the bloom rather than explain "
        "it, and to trust that whatever the garden was working out in its long, patient "
        "arithmetic would eventually arrive at an answer that did not need her help at all."
    ),
    (
        "Almost Always Is Nearly Enough",
        "The clockmaker's sign, hand-painted and slightly crooked, read REPAIRS: ALMOST "
        "ALWAYS SUCCESSFUL, which customers found either reassuring or alarming depending on "
        "how badly they needed their clock fixed. He had chosen the phrasing deliberately, "
        "after a lifetime spent watching other tradesmen promise certainty they could not "
        "deliver. A clock, he had learned, was a small argument between dozens of parts "
        "about what time it currently was, and his job was not to end the argument but to "
        "help the parts reach something close enough to agreement that a person could plan "
        "a train around it. Almost always, the argument settled. Once in perhaps thirty "
        "repairs it did not, and the clock kept some private time of its own no amount of "
        "oil or patience could correct, and on those occasions he handed the clock back "
        "with an honest shrug and no charge, because a broken promise was worse than no "
        "promise at all. His customers, over the years, came to trust the sign more than "
        "they trusted signs that promised everything, and more than one of them said, "
        "picking up a ticking clock at last, that almost always was, in the end, nearly "
        "enough to build a life around."
    ),
    (
        "Everglade",
        "The water moved so slowly through the sawgrass that it seemed, from the airboat, "
        "not to be moving at all, only breathing, rising and falling an inch with the tide "
        "somewhere out past the mangroves. The guide cut the engine in a wide, shallow pool "
        "ringed with cypress and let the silence fill back in, the way it always did within "
        "thirty seconds, birds resuming their arguments, an alligator's eyes surfacing "
        "without a ripple to announce them. He had grown up on this water and still could "
        "not have said, if asked plainly, where the river ended and the swamp began, because "
        "here there was no line, only degrees of wet, only grass giving way to more grass "
        "giving way, eventually, to open sky reflected so perfectly in still water that a "
        "passenger once asked, only half joking, which way was up. He liked that question. "
        "He liked that this place made it a real one. Everything here moved at the speed of "
        "the whole system rather than any one current, and he had come to measure his own "
        "life the same way, less by the single loud events than by the slow, continuous "
        "rise and fall underneath them, the tide he couldn't see doing its patient work "
        "regardless of whether anyone above the grass ever noticed it at all."
    ),
    (
        "Seneca",
        "The letters had been found bundled in oilcloth in the false bottom of an old sea "
        "chest, addressed to a nephew who, as far as anyone could determine, had never "
        "existed. Their author signed himself only as an old man writing to a young one, "
        "and every letter began with the weather and ended with a piece of advice so plain "
        "it seemed almost an insult, until you tried to live by it for a week and discovered "
        "how hard plain advice actually was to follow. Waste no day as though another were "
        "certain to follow it. Own nothing you would grieve to lose more than you would "
        "grieve the grieving itself. The scholar who catalogued the letters for the estate "
        "sale found herself copying passages into her own notebook before she had finished "
        "appraising them, and by the third box she had stopped pretending this was for the "
        "auction catalogue and admitted it was for herself. She never did find the nephew, "
        "real or invented, and came to suspect there had never been one -- that the letters "
        "were a private exercise, a man practising patience on paper because there was no "
        "one left in his life to practise it on aloud, and that she, opening the chest a "
        "hundred years later, had become the nephew he had always been quietly writing to."
    ),
    (
        "Eros",
        "The archaeologists had catalogued four hundred small clay figures from the site, "
        "each one a pair, always a pair, arms reaching toward each other across a gap the "
        "sculptor had left deliberately unfilled, sometimes a hand's width, sometimes barely "
        "a hair's breadth, and no two gaps were exactly the same size. The lead researcher "
        "had a theory she kept mostly to herself, that the gap was the point of the "
        "sculpture and not a flaw in it, that whoever made these figures three thousand "
        "years ago understood something about longing that the finished, touching statues "
        "in the museum upstairs did not: that a hand closing the last inch says less than a "
        "hand permanently, deliberately, an inch away. She had begun, without telling anyone, "
        "measuring the gaps precisely and plotting them against the ages of the burials they "
        "were found beside, and the pattern that was slowly emerging -- larger gaps in "
        "earlier graves, smaller in later ones, as though the culture's patience had worn "
        "down generation by generation -- felt too strange and too human to write into a "
        "formal paper yet, though she suspected, someday, she would have to."
    ),
    (
        "Benway",
        "The consulting physician arrived at the clinic each morning with a leather case "
        "full of instruments no textbook described and a manner so confident that the "
        "nursing staff had long since stopped asking what half of them were for. He spoke "
        "of the human body the way a locksmith speaks of a difficult door: not with awe, "
        "exactly, but with the specific respect of someone who has met every kind of "
        "stubborn mechanism and lost his fear of none of them. His diagnoses arrived fully "
        "formed and slightly askew, correct in outcome and eccentric in reasoning, so that "
        "junior doctors learned to write down his conclusions and quietly discard his "
        "explanations. He kept, in a locked drawer of his desk, a notebook of cases that "
        "official medicine had declined to record, cures that worked for reasons he "
        "admitted, in his own private notation, that he did not entirely understand, "
        "filed not under triumph but under further investigation required, as if certainty "
        "were a luxury his particular corner of the profession could not yet afford."
    ),
    (
        "Onions Wrapped in Rubber",
        "The market stall sold nothing that looked, at first glance, like what it actually "
        "was: apples waxed to mirrors, fish laid on ice carved to resemble more ice, and, "
        "in the far corner, a bin of onions each one individually sheathed in a thin rubber "
        "skin the vendor claimed kept out both frost and the smell that made other vendors' "
        "customers wrinkle their noses. Nobody could say who had started the practice or "
        "why an onion, of all vegetables, had been singled out for this particular armour, "
        "but the rubber skins peeled away cleanly under a thumbnail, and beneath them the "
        "onions kept for months past when an ordinary onion would have gone soft and grey. "
        "A boy who worked the stall on weekends had begun collecting the discarded rubber "
        "skins in a jar, not for any reason he could articulate, simply because they were "
        "beautiful in their own strange way once removed from their purpose -- translucent, "
        "faintly onion-scented, curled into the exact shape of the vegetable they had "
        "protected, small monuments to a job finished and no longer needed."
    ),
    (
        "Tin Cans & Twine",
        "The children had strung the line between two windows across the alley, a can at "
        "each end, and had been assured by an older cousin that this was how telephones "
        "worked before the real ones existed, a claim none of the adults bothered to "
        "correct because it was, in its rough way, true enough. They spoke into the cans "
        "at appointed hours, the twine sagging with morning dew and tightening again in the "
        "afternoon heat, and swore they could tell, from the particular buzz of the string "
        "against their fingertips, whether the other end was smiling. The line survived "
        "three winters, replaced twine twice, and outlasted the friendship it was built for "
        "by exactly one summer, after which one window's can sat silent and swinging while "
        "the other end was taken down and given, eventually, to a younger sibling who never "
        "quite believed a can could carry a voice and spent a whole July trying to prove the "
        "older cousin wrong, whispering increasingly elaborate secrets down an empty length "
        "of string to no one, until, almost by accident, she started answering herself in a "
        "different voice, and found she didn't mind at all."
    ),
    (
        "Spiderwebbed",
        "The old greenhouse had been abandoned for a decade, its panes gone milky with "
        "algae, and in that decade the spiders had built a second roof entirely their own, "
        "web layered over web until the whole glass ceiling shimmered like a second, softer "
        "sky whenever the light caught it right. The new owner, clearing the property for "
        "renovation, stood under it for a long while before touching anything, unwilling to "
        "be the one who tore down ten years of somebody's patient, wordless architecture. "
        "She counted, in that first hour, at least forty distinct webs, some clearly old and "
        "abandoned, threads gone brittle and grey, others fresh enough to still hold morning "
        "dew in perfect beaded lines. She ended up keeping the greenhouse exactly as it was "
        "for another full season, growing nothing in it, visiting most mornings simply to "
        "watch the light come through forty overlapping ceilings instead of one, and only "
        "when the first web-builder's grandchildren, presumably, had inherited the space did "
        "she finally bring in a broom, and even then she worked from the edges inward, "
        "leaving the centre for last, as if giving the newest tenants fair warning."
    ),
    (
        "His Second Story Island",
        "The old man had built the treehouse over two summers with wood scavenged from a "
        "dock that had washed apart in a storm, and he called it, without irony, his second "
        "story island, because from up in the branches the yard below looked exactly like a "
        "coastline he remembered from somewhere he could no longer name precisely. He kept a "
        "single chair up there and nothing else, no books, no radio, just the chair and a "
        "view of a lawn that became, from that height and in that light, something closer to "
        "a shoreline at low tide. His grandchildren visited every summer and each one, in "
        "turn, asked to see the island, and each one, in turn, was disappointed to find only "
        "grass and a garden hose, until he explained, patiently, every time, that an island "
        "wasn't a fact about the ground, it was a fact about the distance you'd travelled to "
        "reach it, and that the real crossing had been the ladder, not the twelve feet of air "
        "beneath it. Most of them forgot the explanation within a week. A few carried it with "
        "them for the rest of their lives, and built their own quiet islands, later, in "
        "attics and back porches and the corners of rooms nobody else wanted."
    ),
    (
        "Cornpone Brunch",
        "Every Sunday without exception, the diner served the same thing regardless of who "
        "asked for what: a plate of cornpone, fried eggs, and whatever vegetable had come in "
        "cheapest that week, and the cook, who had run the place for thirty-one years, "
        "considered the menu a kindness rather than a limitation. Choice, he liked to say to "
        "anyone who complained, was a burden he had decided to carry so his customers didn't "
        "have to, especially on a morning when most of them were nursing a headache from the "
        "night before and in no state to weigh their options. Regulars learned to arrive, "
        "sit, and simply wait, trusting the plate that appeared without being asked the way "
        "they trusted the sunrise, and newcomers who tried to order something else were told, "
        "gently but firmly, that the kitchen only knew how to make one thing well, and that "
        "one thing was, this morning, cornpone. Nobody ever left hungry. Nobody, in thirty-one "
        "years, had ever left disappointed either, which the cook considered proof of a "
        "theory he'd never bothered to write down: that most people didn't actually want more "
        "choices, they wanted one good choice made for them by somebody who cared enough to "
        "get it right every single time."
    ),
    (
        "Salt The Skies",
        "The old fishermen had a saying, passed down without anyone remembering its origin, "
        "that on the clearest nights the stars needed salting the same way a stew did, or "
        "they'd come out flat and tasteless to look at. Nobody took this literally except one "
        "boy, who one August night carried a fistful of coarse salt up to the top of the "
        "seawall and, feeling foolish the whole way, threw it into the wind toward the sky. "
        "It scattered uselessly, of course, most of it blowing straight back into his own "
        "face, and he went to bed certain he'd wasted a good fistful of salt on a saying that "
        "had never meant anything at all. But the next morning, walking the same seawall, he "
        "found grains of it caught in the mortar between the stones, glinting in the early "
        "sun exactly the way stars glinted at night, and he decided, standing there, that the "
        "old fishermen had never meant the sky above at all -- that the salting had already "
        "happened, long before any of them were born, and what they were really teaching, in "
        "their roundabout way, was where else to look for stars once the sky had gone to bed."
    ),
    (
        "The Lithium Stiffs",
        "The old spa had shut its doors decades ago, but the mineral springs still ran "
        "underneath the town, lithium-rich water that the original owners had bottled and "
        "sold as a cure for nerves, melancholy, and, according to one surviving label, "
        "\"stiffness of spirit.\" A new owner reopened the baths with none of the old "
        "claims, just warm water and quiet rooms, but the regulars who came every week swore "
        "the water still did something the plain tap water in their own homes didn't, a "
        "loosening they couldn't describe except to say they walked out straighter than they "
        "walked in, shoulders lower, jaw looser, as if some invisible clenched fist inside "
        "them had, for an hour, agreed to open. The new owner didn't advertise this. She "
        "didn't need to. Word travelled the way it always had in a small town, one loosened "
        "shoulder recommending the water to the next stiff one, and by the second summer she "
        "had a waiting list of people willing to sit quietly in warm mineral water and let "
        "something old and patient underground do, once again, the one thing it had always "
        "known how to do."
    ),
    (
        "High Class Slim Came Floatin' In",
        "Nobody in town could say exactly where he'd come from, only that he arrived on the "
        "afternoon riverboat wearing a suit too fine for the weather and shoes too clean for "
        "the road, and that he moved through the general store like a man who had never once "
        "in his life had to hurry for anything. He bought nothing that first day, only looked, "
        "asking questions in a voice pitched low and pleasant, about the mill, the mayor, the "
        "price of grain, and by suppertime half the town had an opinion about him and none of "
        "the opinions agreed. Some said money. Some said trouble, dressed carefully to look "
        "like money. The old storekeeper, who had seen every kind of stranger come through in "
        "forty years, watched him leave that evening and said only that a man who asked that "
        "many questions and answered none of his own was either buying the town or measuring "
        "it for something else entirely, and that either way, it would be wise to remember "
        "exactly what he'd worn, in case anyone needed to describe him later to somebody who "
        "hadn't been there to see him float in off that boat like he already owned the dock."
    ),
    (
        "Prepare Your Coffin",
        "The old carpenter took commissions for coffins the way other tradesmen took "
        "commissions for furniture, well in advance and with careful attention to the "
        "customer's preferences, and he had long ago stopped finding this strange, though he "
        "understood why new clients often did. Better now than in a hurry later, he told them, "
        "and most, after the first uneasy visit, came to agree: choosing your own wood, your "
        "own joinery, your own lining while you were still healthy enough to have opinions "
        "about it felt less like planning a death and more like commissioning one final, "
        "honest piece of furniture, built exactly the way you wanted rather than however "
        "your grieving family happened to choose in a hurry. He kept finished coffins in a "
        "dry back room, each one labelled discreetly with initials only, and visited them "
        "occasionally the way a farmer walks a field before harvest, checking joints, "
        "rubbing oil into wood that some of his clients had, by now, been sleeping soundly "
        "beside for twenty years and more, patient, prepared, and in absolutely no hurry."
    ),
    (
        "Northern Something",
        "The map the old trapper drew for his grandson had no scale and no compass rose, "
        "only a dotted line running up and off the paper's own edge, labelled in his cramped "
        "handwriting simply NORTH: SOMETHING, WORTH FINDING. He refused, every time asked, to "
        "say what the something was, only that he had seen it twice in sixty years of walking "
        "that country and that both times it had been different, once a light low over a "
        "ridge that moved against the wind instead of with it, once a silence so complete in "
        "the middle of a forest that even his own footsteps had seemed to arrive a half-second "
        "late. He wasn't sure the two things were related. He wasn't sure they weren't. What "
        "he was sure of, and what he tried to pass to his grandson along with the map, was "
        "that the North held things that didn't resolve into explanation no matter how long "
        "you sat with them afterward, and that a person who went looking anyway, accepting in "
        "advance that they might come home with nothing but the going, was the only kind of "
        "person likely to see the something at all."
    ),
    (
        "Gigantes",
        "The quarry workers found the bones by accident, widening a road cut into a hillside "
        "that had, according to every local story, always been unlucky ground. Vertebrae the "
        "size of dinner plates, a femur too long for any two workers to lift together, and no "
        "agreement among the scholars who eventually arrived about what animal, ancient or "
        "otherwise, could have carried such a frame. The oldest villagers were unsurprised. "
        "Their grandmothers had always said giants slept under that hill, not dead exactly but "
        "resting, the way a mountain rests, on a timescale that made a human lifetime look like "
        "a held breath. The scholars measured, photographed, and eventually crated the bones "
        "for a museum three provinces away, and the hill, once excavated, settled back into "
        "ordinary quiet, grass growing over the cut within two seasons. But the villagers kept "
        "leaving small offerings at the roadside anyway, bread mostly, sometimes a coin, on the "
        "theory that a resting giant, disturbed enough to be measured and carted off in pieces, "
        "deserved at least the courtesy of a gift left behind for whatever of it remained."
    ),
    (
        "Yinxianghechengqi",
        "The instrument had no name anyone in the workshop recognised, only a set of "
        "characters stamped into its brass base that none of the current craftsmen could "
        "read, though every one of them agreed it was beautifully made: a lattice of gears "
        "and tuned rods that, when wound, produced not a melody exactly but a layered "
        "resonance, several tones arriving and departing at their own separate speeds, so "
        "that no two windings ever sounded quite the same. The old master who had trained "
        "them all claimed it had come from a workshop that no longer existed, in a language "
        "that had drifted since, and that its maker had built it not to be understood but to "
        "be listened to, which he insisted were two entirely different skills. He would not "
        "let anyone take it apart to see how it worked. Understanding it, he said, would ruin "
        "the specific pleasure of not understanding it, of winding the key and simply "
        "receiving whatever layered, untranslatable answer the machine chose to give that "
        "day, the way you receive weather, or grief, or any other honest thing too large to "
        "be reduced to a single, satisfying explanation."
    ),
    (
        "The Fall of Seven Diamonds Plus One",
        "The card game had been played in that back room for as long as anyone could "
        "remember, by rules nobody had ever written down and nobody entirely agreed on, and "
        "its one unbreakable law concerned the seven diamonds: whenever all seven fell from a "
        "single hand in one round, plus any eighth card of no fixed suit, the table went "
        "silent and the dealer, whoever it happened to be that night, had to stand and recite "
        "a short toast to whoever had last held that particular fall before losing everything "
        "on the hand after it. Nobody remembered who that unlucky first player had been. The "
        "toast had been passed down anyway, word for word, through forty years of Tuesday "
        "nights, a ritual observed maybe once a decade when the cards happened to fall that "
        "exact way, and every player who ever witnessed it agreed on one thing afterward: for "
        "the length of that one toast, win or lose meant nothing at all, and the whole room "
        "was simply, briefly, remembering someone none of them had ever met."
    ),
    (
        "Monument Six One Thousand",
        "The surveyor's marker was one of a series driven into the ground a century earlier, "
        "numbered in a scheme that made sense only to the long-retired office that had "
        "commissioned it, and this one, stamped 6-1000 on its weathered brass cap, sat alone "
        "in a field that had reverted to scrub since the original property lines dissolved "
        "into a dozen smaller ones and then, eventually, into nothing anyone bothered to "
        "fence. A retired schoolteacher had taken up, in her later years, the quiet hobby of "
        "finding every marker in the old numbering scheme and photographing it exactly as it "
        "stood, weeds and all, cataloguing a system that no longer governed anything, simply "
        "because someone, once, had thought it worth marking this precise spot on the earth "
        "and she found she couldn't bear the thought of that care going entirely unwitnessed. "
        "She had found forty-one of an estimated sixty markers by the time this one turned up, "
        "half-buried and tilted, and she knelt in the wet grass a long while before "
        "photographing it, less out of reverence for the marker itself than for the patient, "
        "anonymous certainty of whoever had once needed the world divided this exactly."
    ),
    (
        "De Chelly",
        "The canyon walls held petroglyphs at heights no visible ladder could explain, "
        "figures of long-limbed animals and spirals that meant nothing to the ranger who "
        "led tours through in summer and everything, she suspected, to whoever had carved "
        "them, standing on ground that had since eroded away entirely, leaving their work "
        "stranded thirty feet above the current canyon floor. She had stopped trying to "
        "explain the height to tourists years ago; the honest answer was that the canyon "
        "itself had changed since the carving, sand and rock worn down around a stillness "
        "that hadn't moved at all, and that the strangeness people felt looking up wasn't a "
        "mystery about ladders, it was a fact about time, made suddenly, physically visible. "
        "She liked bringing groups through at the end of the day, when the low sun turned the "
        "canyon walls the colour of embers and the petroglyphs seemed, briefly, to shift with "
        "the changing light the way a fire shifts, and she would tell them, every time, that "
        "the figures weren't stranded at all. The canyon floor was the thing that had moved. "
        "The carvings had simply stayed exactly where their makers had always meant them to be."
    ),
    (
        "Gesceap",
        "The old scribe's marginal note, in a hand three centuries older than the manuscript "
        "it annotated, used a single word that the modern translator had spent a full week "
        "trying to render correctly: gesceap, meaning at once a thing created, the shape it "
        "was created into, and the fate that shape implied, three ideas the old language had "
        "never bothered separating because it had never needed to. The translator's working "
        "draft carried the word in six different English attempts, each one catching part of "
        "it and losing the rest, and she had begun to suspect that the old scribe, writing in "
        "the margin of somebody else's story, had chosen this exact word precisely because it "
        "refused a clean translation, because the maker, the made thing, and the destiny of "
        "the made thing were, in his understanding of the world, never actually three things "
        "at all. She left the word untranslated in her final draft, a small island of the "
        "original language sitting in the middle of the English page, with a footnote longer "
        "than the sentence it explained, and felt, submitting the manuscript at last, that "
        "some words earned the right to stay exactly what they already were."
    ),
]


def iter_tortoise_rows() -> Iterator[Dict[str, object]]:
    """Yield each vignette as a ``{"text": ...}`` row, in registration order."""
    for _title, vignette in TORTOISE_VIGNETTES:
        yield {"text": vignette}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None,
                   help="Override the destination path (default: artifacts/raw/tortoise/text.jsonl).")
    args = p.parse_args()

    dest = args.out or (shared_dir("raw") / "tortoise" / "text.jsonl")
    n = write_documents(iter_tortoise_rows(), dest)
    print(f"tortoise: {n:,} documents -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fetch_tortoise.py -q`
Expected: PASS, all 6 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_tortoise.py tests/test_fetch_tortoise.py
git commit -m "feat: add the hand-authored tortoise vignette corpus (30 verified titles)"
```

---

## Task 6: Frozen eval prompt set C

**Files:**
- Create: `docs/evaluation_prompts_c.json`
- Create: `tests/test_evaluation_prompts_c.py`

**Interfaces:**
- Consumes: `docs/evaluation_prompts.json`, `docs/evaluation_prompts_b.json` (existing, for
  the disjointness tests).
- Produces: a third frozen prompt set, `docs/evaluation_prompts_c.json`, for the follow-up
  training-run plan's Gate 6.

Thirty prompts, one per `TORTOISE_VIGNETTES` title (Task 5), each rendered as a short scene
opening rather than the bare title — the instrument that actually measures whether the
trained model "earns its namesake" by producing evocative, coherent continuations, as
distinct from the tiny corpus slice's share (which cannot move an aggregate metric at this
size). Mirrors `evaluation_prompts_b.json`'s exact schema and test pattern.

- [ ] **Step 1: Write `docs/evaluation_prompts_c.json`**

```json
{
  "note": "Frozen evaluation set C. A THIRD frozen set, built from real Tortoise (the band) instrumental track titles -- see docs/superpowers/specs/2026-09-28-storytelling-register-stage-design.md. Every prompt seeds a scene toward the imagery of one verified title without quoting it directly. This is the instrument that measures whether the model produces evocative, coherent continuations from an abstract prompt, as distinct from the tiny 'tortoise' corpus slice (train/corpus.py), which cannot move an aggregate metric at this scale on its own. NEVER pooled with sets A or B: three sets written at different times, for different purposes, are not exchangeable. Do not edit prompts between runs -- that breaks comparability. Add new ones with new ids instead.",
  "design": [
    "Every id is prefixed 'c-tortoise-', disjoint by construction from set A's ('voice-', 'stutter-', 'oracle-', 'agentic-', 'ground-', 'assoc-', 'long-') and set B's ('b-').",
    "Every prompt carries the single probe tag 'evocative-continuation' -- unlike sets A and B, which spread across several named behaviours, set C is deliberately single-purpose: it exists to answer one question (does the model go somewhere interesting from an abstract seed, or default to a TinyStories attractor), not to decompose that question into sub-behaviours.",
    "Prompts are ordered to match TORTOISE_VIGNETTES (scripts/fetch_tortoise.py) exactly, numbered 01 through 30, so a reader can trace any prompt back to the real title and album it was seeded from without re-deriving the mapping.",
    "Each prompt is a short scene opening that evokes the title's imagery without quoting the title itself verbatim inside the prompt text, so scoring the completion for register/collapse markers is not contaminated by the prompt echoing the title back.",
    "This set is scored with the exact same instruments (scripts/score_behaviour.py, scripts/evaluate.py) as sets A and B -- no new detector was built for it. A qualitative human read (Gate 6 in the training-run plan) is the primary evidence this set is FOR; the quantitative collapse-rate signals are secondary corroboration, not the headline."
  ],
  "prompts": [
    {"id": "c-tortoise-01", "probe": "evocative-continuation",
     "text": "The apprentice built the device from copper wire and an old steam pipe, and on the appointed evening he stood inside the ring of chalk and switched it"},
    {"id": "c-tortoise-02", "probe": "evocative-continuation",
     "text": "The pillar had stood in the museum's basement for forty years, uncatalogued, until a guide searching for a spare projector bulb ran her hand along it and felt"},
    {"id": "c-tortoise-03", "probe": "evocative-continuation",
     "text": "Every case in the collection held something that used to hold something else, and on quiet nights the whole room produced a low,"},
    {"id": "c-tortoise-04", "probe": "evocative-continuation",
     "text": "The boy climbed because the roofline made more sense to him than the street did, hand over hand along the gutters, which groaned under his weight but"},
    {"id": "c-tortoise-05", "probe": "evocative-continuation",
     "text": "Every ten days, without fail, a letter arrived at the lighthouse addressed to no one in particular, describing weather that had not yet"},
    {"id": "c-tortoise-06", "probe": "evocative-continuation",
     "text": "She turned away from the harbour lights on purpose, because the harbour was where everyone else was looking, and set her face toward the"},
    {"id": "c-tortoise-07", "probe": "evocative-continuation",
     "text": "The old surveyor kept a brass rod exactly one metre long and swore he could feel the line itself underfoot whenever he crossed"},
    {"id": "c-tortoise-08", "probe": "evocative-continuation",
     "text": "The bridge had been built to survive the spray, and a guide walked it every morning before the gates opened, alone, listening for"},
    {"id": "c-tortoise-09", "probe": "evocative-continuation",
     "text": "The garden kept its own schedule, four days between one strange bloom and the next, and the old woman who tended it had stopped"},
    {"id": "c-tortoise-10", "probe": "evocative-continuation",
     "text": "The clockmaker's sign, hand-painted and slightly crooked, made a promise that customers found either reassuring or alarming, depending on"},
    {"id": "c-tortoise-11", "probe": "evocative-continuation",
     "text": "The water moved so slowly through the sawgrass that it seemed, from the boat, not to be moving at all, only"},
    {"id": "c-tortoise-12", "probe": "evocative-continuation",
     "text": "The letters had been found bundled in oilcloth in the false bottom of an old sea chest, addressed to a nephew who, as far as anyone could tell,"},
    {"id": "c-tortoise-13", "probe": "evocative-continuation",
     "text": "The archaeologists had catalogued four hundred small clay figures from the site, each one a pair, arms reaching across a gap the sculptor had left"},
    {"id": "c-tortoise-14", "probe": "evocative-continuation",
     "text": "The consulting physician arrived at the clinic each morning with a leather case full of instruments no textbook described, and spoke of the body as"},
    {"id": "c-tortoise-15", "probe": "evocative-continuation",
     "text": "The market stall sold nothing that looked, at first glance, like what it actually was, and in the far corner sat a bin of onions each one"},
    {"id": "c-tortoise-16", "probe": "evocative-continuation",
     "text": "The children had strung the line between two windows across the alley, a can at each end, and spoke into it at appointed hours, certain they could tell"},
    {"id": "c-tortoise-17", "probe": "evocative-continuation",
     "text": "The old greenhouse had been abandoned for a decade, and in that decade the spiders had built a second roof entirely their own, layered until"},
    {"id": "c-tortoise-18", "probe": "evocative-continuation",
     "text": "The old man had built the treehouse over two summers, and called it his second story island, because from up in the branches the yard below looked exactly like"},
    {"id": "c-tortoise-19", "probe": "evocative-continuation",
     "text": "Every Sunday without exception, the diner served the same plate regardless of who asked for what, and the cook considered the menu"},
    {"id": "c-tortoise-20", "probe": "evocative-continuation",
     "text": "The old fishermen had a saying, passed down without anyone remembering its origin, that on the clearest nights the stars needed"},
    {"id": "c-tortoise-21", "probe": "evocative-continuation",
     "text": "The old spa had shut its doors decades ago, but the mineral springs still ran underneath the town, and the regulars who came every week swore the water still"},
    {"id": "c-tortoise-22", "probe": "evocative-continuation",
     "text": "Nobody in town could say exactly where he'd come from, only that he arrived on the afternoon riverboat wearing a suit too fine for the weather and"},
    {"id": "c-tortoise-23", "probe": "evocative-continuation",
     "text": "The old carpenter took commissions for coffins the way other tradesmen took commissions for furniture, and told his clients, better now than"},
    {"id": "c-tortoise-24", "probe": "evocative-continuation",
     "text": "The map the old trapper drew for his grandson had no scale and no compass rose, only a dotted line running north, labelled"},
    {"id": "c-tortoise-25", "probe": "evocative-continuation",
     "text": "The quarry workers found the bones by accident, widening a road cut into a hillside that had, according to every local story,"},
    {"id": "c-tortoise-26", "probe": "evocative-continuation",
     "text": "The instrument had no name anyone in the workshop recognised, only a set of characters stamped into its brass base, and when wound it produced"},
    {"id": "c-tortoise-27", "probe": "evocative-continuation",
     "text": "The card game had been played in that back room for as long as anyone could remember, and its one unbreakable law concerned the night all seven"},
    {"id": "c-tortoise-28", "probe": "evocative-continuation",
     "text": "The surveyor's marker was one of a series driven into the ground a century earlier, and a retired schoolteacher had taken up the quiet hobby of"},
    {"id": "c-tortoise-29", "probe": "evocative-continuation",
     "text": "The canyon walls held carvings at heights no visible ladder could explain, and the ranger who led tours through in summer had stopped trying to"},
    {"id": "c-tortoise-30", "probe": "evocative-continuation",
     "text": "The old scribe's marginal note used a single word that the modern translator had spent a full week trying to render, a word that meant at once a thing created and"}
  ]
}
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_evaluation_prompts_c.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Prompt set C is FROZEN, exactly as sets A and B are."""
import hashlib
import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
PROMPTS_C = DOCS / "evaluation_prompts_c.json"
PROMPTS_A = DOCS / "evaluation_prompts.json"
PROMPTS_B = DOCS / "evaluation_prompts_b.json"

ID_PREFIX = "c-tortoise-"
FROZEN_COUNT = 30
#: Computed from the prompts embedded above once written; see Step 3.
FROZEN_DIGEST = "PLACEHOLDER_COMPUTED_IN_STEP_3"


def _digest(prompts) -> str:
    h = hashlib.sha256()
    for pid, text in sorted((p["id"], p["text"]) for p in prompts):
        h.update(pid.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _prompts_c():
    return json.loads(PROMPTS_C.read_text())["prompts"]


def _prompts_a():
    return json.loads(PROMPTS_A.read_text())["prompts"]


def _prompts_b():
    return json.loads(PROMPTS_B.read_text())["prompts"]


def test_prompt_file_parses_and_carries_its_design():
    data = json.loads(PROMPTS_C.read_text())
    assert data["prompts"]
    assert data["note"].strip()
    assert isinstance(data["design"], list) and len(data["design"]) >= 4
    for entry in data["design"]:
        assert entry.strip()


def test_every_prompt_has_the_same_schema_as_set_a_and_b():
    keys_a = {frozenset(p) for p in _prompts_a()}
    keys_c = {frozenset(p) for p in _prompts_c()}
    assert keys_c == keys_a == {frozenset({"id", "probe", "text"})}


def test_ids_are_unique():
    ids = [p["id"] for p in _prompts_c()]
    assert len(ids) == len(set(ids))


def test_every_id_uses_the_c_prefix_and_cannot_collide_with_a_or_b():
    ids_c = [p["id"] for p in _prompts_c()]
    ids_a = [p["id"] for p in _prompts_a()]
    ids_b = [p["id"] for p in _prompts_b()]
    assert all(pid.startswith(ID_PREFIX) for pid in ids_c)
    assert not any(pid.startswith(ID_PREFIX) for pid in ids_a + ids_b)
    assert not (set(ids_a) & set(ids_c))
    assert not (set(ids_b) & set(ids_c))


def test_every_prompt_uses_the_single_evocative_continuation_probe():
    """Unlike sets A and B, set C is deliberately single-purpose."""
    probes = {p["probe"] for p in _prompts_c()}
    assert probes == {"evocative-continuation"}


def test_no_prompt_text_is_shared_with_set_a_or_b():
    texts_c = {p["text"] for p in _prompts_c()}
    assert not (texts_c & {p["text"] for p in _prompts_a()})
    assert not (texts_c & {p["text"] for p in _prompts_b()})


def test_no_prompt_is_empty_or_whitespace():
    for p in _prompts_c():
        assert p["text"].strip()


def test_no_prompt_text_is_duplicated_within_the_set():
    texts = [p["text"] for p in _prompts_c()]
    assert len(texts) == len(set(texts))


def test_no_prompt_quotes_its_own_seed_title_verbatim():
    """Scoring a completion for register/collapse markers should not be contaminated by
    the prompt itself echoing the title back -- see scripts/fetch_tortoise.py for the
    real titles this set was seeded from."""
    from scripts.fetch_tortoise import TORTOISE_VIGNETTES

    for (title, _), prompt in zip(TORTOISE_VIGNETTES, _prompts_c()):
        assert title.lower() not in prompt["text"].lower(), (
            f"{prompt['id']!r} quotes its own seed title {title!r} verbatim"
        )


def test_prompt_text_is_frozen_not_just_the_ids():
    prompts = _prompts_c()
    assert len(prompts) == FROZEN_COUNT
    assert _digest(prompts) == FROZEN_DIGEST, (
        "prompt set C changed. Samples generated before this edit are no longer "
        "comparable with ones generated after it. If that is intended, update "
        "FROZEN_DIGEST and FROZEN_COUNT in the same commit and say why.")


def test_the_digest_actually_detects_a_rewritten_prompt():
    tampered = [dict(p) for p in _prompts_c()]
    tampered[0]["text"] = tampered[0]["text"] + " and then everything changed"
    assert _digest(tampered) != FROZEN_DIGEST


def test_the_digest_detects_a_prompt_being_dropped_or_added():
    prompts = _prompts_c()
    assert _digest(prompts[:-1]) != FROZEN_DIGEST
    assert _digest(prompts + [{"id": "c-tortoise-31", "text": "one more", "probe": "evocative-continuation"}]) != FROZEN_DIGEST


def test_the_digest_detects_text_moving_between_two_prompts():
    tampered = [dict(p) for p in _prompts_c()]
    tampered[0]["text"], tampered[1]["text"] = tampered[1]["text"], tampered[0]["text"]
    assert _digest(tampered) != FROZEN_DIGEST


def test_the_digest_does_not_depend_on_file_order():
    assert _digest(list(reversed(_prompts_c()))) == FROZEN_DIGEST


def test_set_c_does_not_share_a_digest_with_a_or_b():
    assert _digest(_prompts_c()) != _digest(_prompts_a())
    assert _digest(_prompts_c()) != _digest(_prompts_b())
```

- [ ] **Step 3: Compute the real digest and fix the placeholder**

Run:

```bash
python3 -c "
import hashlib, json
prompts = json.loads(open('docs/evaluation_prompts_c.json').read())['prompts']
h = hashlib.sha256()
for pid, text in sorted((p['id'], p['text']) for p in prompts):
    h.update(pid.encode('utf-8')); h.update(b'\x00')
    h.update(text.encode('utf-8')); h.update(b'\x00')
print(h.hexdigest())
"
```

Replace `FROZEN_DIGEST = "PLACEHOLDER_COMPUTED_IN_STEP_3"` in the test file with the real
printed hex digest.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluation_prompts_c.py -q`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add docs/evaluation_prompts_c.json tests/test_evaluation_prompts_c.py
git commit -m "feat: add frozen evaluation prompt set C (30 Tortoise-title seeds)"
```

---

## Task 7: Register `pg19` and `tortoise`, reweight every share

**Files:**
- Modify: `train/corpus.py` (`SOURCES` — two new entries, every existing entry's
  `target_share` updated)
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `CorpusSource.words_per_document` (Task 2), `scripts.fetch_pg19.PG19_REVISION`
  (Task 3), `scripts.fetch_tortoise` (Task 5).
- Produces: `SOURCES["pg19"]`, `SOURCES["tortoise"]`, and a full registry summing to exactly
  1.0. Task 8 fetches against these entries for real.

First-draft shares, to be settled for real against measured availability in Task 9 — this
project's own established pattern (`target_share` fields are documented as targets, not
measurements). `tortoise` starts at `0.0`, matching the `pulp_sf` precedent for "registered,
not yet populated": its real ceiling is measured and settled in Task 9, the same way
`flavour`'s ceiling was originally found.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_corpus.py

def test_pg19_and_tortoise_are_registered_with_correct_fetch_kinds():
    from train.corpus import SOURCES

    pg19 = SOURCES["pg19"]
    assert pg19.fetch_kind == "url"
    assert pg19.words_per_document == 3000
    assert pg19.rows_per_document == 1
    assert "Apache-2.0" in pg19.license_id
    assert "public domain" in pg19.license_note.lower()

    tortoise = SOURCES["tortoise"]
    assert tortoise.fetch_kind == "url"
    assert tortoise.target_share == 0.0  # unsettled, matching pulp_sf's precedent


def test_tinystories_share_was_cut_further():
    """Continuing the one intervention already proven to move register."""
    from train.corpus import SOURCES

    assert SOURCES["tinystories"].target_share < 0.10
```

(`test_target_shares_sum_to_one`, already in this file, is the load-bearing check for the
reweight — no new test is needed for it, it just has to keep passing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_corpus.py -k "pg19_and_tortoise or tinystories_share_was_cut" -q`
Expected: FAIL — `KeyError: 'pg19'`.

- [ ] **Step 3: Add the two new entries and reweight every existing share**

In `train/corpus.py`, change `tinystories`'s `target_share` from `0.10` to `0.030`, add a
line to its `rationale` noting the further cut and pointing at this plan's spec. Change every
other existing source's `target_share` to the values below (first-draft numbers, to be
settled for real in Task 9), leaving every other field on each entry untouched:

| source | old share | new share |
|---|---:|---:|
| `dialogue` | 0.020 | 0.020 (unchanged) |
| `tinystories` | 0.10 | 0.030 |
| `gutenberg_children` | 0.15 | 0.030 |
| `wikipedia_simple` | 0.15 | 0.030 |
| `spine` | 0.135 | 0.025 |
| `folklore` | 0.08 | 0.015 |
| `weird` | 0.04 | 0.008 |
| `poetry` | 0.01 | 0.002 |
| `procedural` | 0.12 | 0.024 |
| `flavour` | 0.005 | 0.001 |
| `longform` | 0.188 | 0.250 |
| `mission` | 0.002 | 0.001 |
| `pulp_sf` | 0.0 | 0.0 (unchanged) |
| `pg19` (new) | — | 0.564 |
| `tortoise` (new) | — | 0.0 |

Add a one-line rationale suffix to each reweighted entry's existing `rationale` string
(e.g. `" Share cut from 0.15 to 0.030 on 2026-09-28 to fund pg19 -- see "
"docs/superpowers/specs/2026-09-28-storytelling-register-stage-design.md. First-draft "
"number, to be settled against real measured availability."`), so the registry's own
history stays legible the way every prior resettlement in this file already is. Every
other field on each entry (`hf_repo`, `hf_revision`, `authors`, `bookshelves`, `upsample`,
`slice`, etc.) is untouched — only `target_share` and `rationale` change. Worked example
for `tinystories` (apply the same shape of edit to every other row in the table above,
substituting its own old/new values):

```python
    "tinystories": CorpusSource(
        name="tinystories",
        slice="backbone",
        target_share=0.030,
        hf_repo="roneneldan/TinyStories",
        hf_revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        license_id="CDLA-Sharing-1.0",
        license_url="https://cdla.dev/sharing-1-0/",
        attribution="TinyStories (Eldan & Li), roneneldan/TinyStories",
        share_alike=True,
        rationale="Simple, regular grammar. The backbone that makes a small model readable. "
                  # ... existing rationale text, unchanged, up to its final period ...
                  "needing 0.2768x. Share cut further from 0.10 to 0.030 on 2026-09-28 to "
                  "fund pg19 -- see "
                  "docs/superpowers/specs/2026-09-28-storytelling-register-stage-design.md. "
                  "Continues the one intervention already proven to move register (the "
                  "2026-08-27 10% arm). First-draft number, to be settled against real "
                  "measured availability in Task 9.",
    ),
```

Add the two new entries:

```python
    "pg19": CorpusSource(
        name="pg19",
        slice="backbone",
        target_share=0.564,
        hf_repo="", hf_revision="",
        fetch_kind="url",
        # A representative anchor for the "resolvable fetch spec" shape every url source
        # declares -- the real fetch spec is scripts/fetch_pg19.py's per-book URL
        # construction (28,602 books), not this single URL. Never itself fetched.
        source_url="https://storage.googleapis.com/deepmind-gutenberg/train/10.txt",
        license_id="Apache-2.0 (packaging); public domain (texts, pre-1919)",
        license_url="https://huggingface.co/datasets/deepmind/pg19",
        attribution="PG-19 (Rae et al., DeepMind), deepmind/pg19",
        license_note=(
            "Apache-2.0 covers DeepMind's packaging of the dataset. The underlying texts "
            "are Project Gutenberg books published before 1919 and are public domain by "
            "age -- the same dual structure already carried by gutenberg_children/spine/"
            "folklore/weird/procedural/flavour (MIT packaging there, instead of "
            "Apache-2.0). Verified via the dataset card and its loading script directly "
            "(scripts/fetch_pg19.py's module docstring), not assumed from a summary. "
            "PG-19's own dataset card explicitly cautions against training a "
            "general-purpose language model on it alone, citing dated linguistic style "
            "and the biases of historical writing -- recorded here rather than only in "
            "the spec, since the caution comes from the dataset's own authors."
        ),
        rows_per_document=1,
        words_per_document=3000,
        rationale=(
            "The register/vocabulary-richness source this plan exists to add -- see "
            "docs/superpowers/specs/2026-09-28-storytelling-register-stage-design.md. "
            "Full novels, not TinyStories-scale short stories, so words_per_document=3000 "
            "splits each book into paragraph-aligned sub-documents rather than writing a "
            "whole novel as one document with a single, near-useless closing separator. "
            "First-draft share; settled against real measured availability in the "
            "corpus-build task that follows this registration."
        ),
    ),
    "tortoise": CorpusSource(
        name="tortoise",
        slice="flavour",
        target_share=0.0,
        hf_repo="", hf_revision="",
        fetch_kind="url",
        # A representative anchor, never itself fetched -- the real content is
        # hand-authored in scripts/fetch_tortoise.py, the same pattern mission and
        # pulp_sf already use for a source whose content lives in a dedicated script.
        source_url="https://en.wikipedia.org/wiki/Tortoise_(band)",
        license_id="",
        license_url="",
        attribution="Original vignettes written for this project, seeded by real Tortoise "
                    "(band) instrumental track titles",
        license_note=(
            "Song and album TITLES are short phrases, not independently copyrightable "
            "text on their own; the vignettes themselves are original text written for "
            "this project, not song lyrics or descriptions of the recordings. No third-"
            "party licence is implicated. Verified real, not invented: every title in "
            "scripts/fetch_tortoise.py::TORTOISE_VIGNETTES was cross-checked against the "
            "band's discography (Wikipedia, Discogs, AllMusic) before use."
        ),
        rows_per_document=1,
        rationale=(
            "Small, hand-authored flavour-class register injection -- see the spec's "
            "'flavour precedent' framing. Registered at target_share=0.0, matching "
            "pulp_sf's 'registered, not yet populated' pattern, because its real ceiling "
            "is not yet measured; settled in the corpus-build task that follows this "
            "registration, expected (per the spec) to land well under 0.5%, the same "
            "order of magnitude as flavour's own settled share."
        ),
    ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_corpus.py -q`
Expected: PASS, including `test_target_shares_sum_to_one` and both new tests.

- [ ] **Step 5: Commit**

```bash
git add train/corpus.py tests/test_corpus.py
git commit -m "feat: register pg19 and tortoise sources, reweight the blend (first draft)"
```

---

## Task 8: Real fetch of every source, and the Gate 3 measurement

**Files:**
- No new source files; this task runs the pipeline against real network access and produces
  `docs/measurements/corpus_availability.json` (tracked).

**Interfaces:**
- Consumes: every source registered in Task 7, `scripts/fetch_corpus.py`,
  `scripts/fetch_pg19.py`, `scripts/fetch_tortoise.py`, `scripts/prepare_corpus.py`,
  `scripts/measure_corpus.py`.
- Produces: `artifacts/raw/<name>/text.jsonl` and `artifacts/corpus/<name>.txt` per source
  (gitignored), `docs/measurements/corpus_availability.json` (tracked, real numbers).

This is real execution against real network data, not TDD code — the deliverable is a
measured, real availability report Task 9 settles shares against.

- [ ] **Step 1: Fetch every "hf"-kind source**

Run: `python scripts/fetch_corpus.py`
Expected: one line per source (`dialogue`, `tinystories`, `gutenberg_children`,
`wikipedia_simple`, `spine`, `folklore`, `weird`, `poetry`, `procedural`, `flavour`,
`longform`), each reporting a nonzero document count. `pulp_sf` and the two new
`fetch_kind="url"` sources (`pg19`, `tortoise`) are NOT fetched by this script — confirm the
existing `main()` dispatch logic (Gutenberg batch + per-source loop) does not attempt to fetch
them, since their `hf_repo` is empty.

- [ ] **Step 2: Fetch `mission`, `pg19`, and `tortoise` separately**

Run:
```bash
python scripts/fetch_mission.py
python scripts/fetch_pg19.py --limit-books 20000
python scripts/fetch_tortoise.py
```

`--limit-books 20000` caps PG-19 well above what a ~0.56 share of a 2.5B budget needs (the
first-draft share implies roughly 1.41B tokens; PG-19 book text tokenizes close to 1
token/word, so 20,000 books — a large majority of the available 28,602 — gives comfortable
headroom without downloading the entire corpus). Expected: `mission: 1 documents`,
`pg19: <=20,000 documents` (some books may fetch empty and be skipped — a handful of
warnings is expected and fine), `tortoise: 30 documents`.

- [ ] **Step 3: Prepare every source**

Run: `python scripts/prepare_corpus.py`
Expected: one line per source under `sorted(SOURCES)` with a nonzero row-to-document count.
For `pg19` specifically, confirm `documents` is much larger than the number of fetched books
(each book split into several ~3000-word sub-documents by Task 2's chunking) — e.g. a
20,000-book fetch producing document count noticeably above 20,000, not equal to it (equal
would mean the chunking silently did nothing).

- [ ] **Step 4: Run the Gate 3 measurement**

Run: `python scripts/measure_corpus.py --budget 2500000000 --upsample-cap 8`
Expected output includes a gate table for all 15 sources and either "All slices can reach
their target share within the cap." (exit 0) or a shortfall report (exit 1). Either outcome
is a valid result of this step — Task 9 is where the shortfall report (if any) gets acted on.
This overwrites `docs/measurements/corpus_availability.json` with the real measurement;
confirm it is tracked (`git status` shows it modified, not gitignored).

- [ ] **Step 5: Commit the real measurement**

```bash
git add docs/measurements/corpus_availability.json
git commit -m "data: real corpus availability measurement against the reweighted blend"
```

---

## Task 9: Settle real shares, regenerate licensing, update the README

**Files:**
- Modify: `train/corpus.py` (real, measurement-backed `target_share` values and
  `rationale` updates)
- Modify: `docs/corpus_licensing.md` (regenerated, not hand-edited)
- Modify: `README.md` (Provenance and licensing section)
- Test: `tests/test_corpus.py` (must still pass; no new tests — this task revises data, not
  behaviour)

**Interfaces:**
- Consumes: `docs/measurements/corpus_availability.json` (Task 8's real numbers).
- Produces: a registry whose shares are backed by real measurement, and provenance
  documentation that matches it.

This is where Task 7's first-draft shares become real, following the exact established
pattern (`flavour`, `spine`, `procedural`'s own rationale text all document this same kind of
settle-against-measurement revision).

- [ ] **Step 1: Read the Gate 3 report from Task 8**

Open `docs/measurements/corpus_availability.json`. For any source in `shortfalls`, its
`needed_upsample` exceeds `upsample_cap` (8) at its current `target_share` — this needs a
share reduction, following the exact escalation this project always uses: reduce the
tightest slice's share, redistribute the freed share toward whichever slice has the most
measured headroom (per this file's own extensive precedent in `spine`'s and `procedural`'s
rationale text). If nothing is in `shortfalls`, this step is "confirm no action needed" and
move directly to Step 2.

- [ ] **Step 2: Update every touched source's `target_share` and `rationale`**

Edit `train/corpus.py` with the real, settled numbers. Every source whose share changes from
its Task 7 first-draft value gets a rationale addendum stating the old value, the new value,
and the measured number that drove the change — matching every existing entry's own style
exactly (e.g. `flavour`'s multi-sentence history of exactly this kind of revision).

Run `python -m pytest tests/test_corpus.py -q` after every edit — `test_target_shares_sum_to_one`
must pass before moving on.

- [ ] **Step 3: Regenerate the licensing document**

Run: `python scripts/render_licensing.py`
Expected: `docs/corpus_licensing.md` is rewritten in place, now including `pg19` and
`tortoise` entries with their real, measured shares.

- [ ] **Step 4: Update `README.md`'s "Provenance and licensing" section**

Read the current section (`README.md`, `## Provenance and licensing`) before editing —
it already contains at least one stale figure predating this plan (a `longform` share
description that does not match the live registry), so do not copy an existing sentence's
numbers without checking them against `train/corpus.py` directly. Add a paragraph describing
`pg19` (Apache-2.0 packaging / public-domain texts, its real settled share, and the dataset
card's own caution about training a general-purpose model on it alone) and `tortoise`
(no third-party licence implicated, titles verified real) in the same style as the existing
`longform` paragraph, with the REAL settled shares from Step 2 — not the first-draft numbers
from Task 7.

- [ ] **Step 5: Run tests to verify nothing broke**

Run: `python -m pytest tests/test_corpus.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add train/corpus.py docs/corpus_licensing.md README.md
git commit -m "docs: settle real corpus shares against measurement, regenerate provenance"
```

---

## Task 10: Build the final blend, retrain the tokenizer (Gates 1–2)

**Files:**
- No new source files; produces `artifacts/corpus/blend.txt`, `artifacts/tokenizer`
  (overwritten in place), `docs/measurements/blend_manifest.json` (tracked).

**Interfaces:**
- Consumes: the settled registry (Task 9), `scripts/blend_corpus.py`,
  `scripts/build_tokenizer.py`.
- Produces: the final blend and tokenizer Task 11 tokenizes against.

- [ ] **Step 1: Build the blend**

Run: `python scripts/blend_corpus.py --budget 2500000000`
Expected: a per-source table, a total near 2,500,000,000 tokens (small variance from
truncation is normal and expected, matching every prior blend's own `total_vs_budget_pct`),
`artifacts/corpus/blend.txt` written, and `docs/measurements/blend_manifest.json` written
both under `artifacts/corpus/` and to the tracked default
(`docs/measurements/blend_manifest.json`).

Note: this first pass runs with whatever tokenizer currently exists at `artifacts/tokenizer`
(the old 32K one) — `blend_corpus.py`'s `TokenMeter` falls back to an approximation if no
tokenizer is found, so this pass measures blend composition, not final token counts; Step 3
retrains the tokenizer and the blend does not need rebuilding afterward, since `blend.txt` is
plain text independent of which tokenizer later reads it.

- [ ] **Step 2: Gate 1 — separator density**

Run:
```bash
python3 -c "
from pathlib import Path
text = Path('artifacts/corpus/blend.txt').read_text(encoding='utf-8', errors='replace')
seps = text.count('</s>')
tokens_approx = len(text.split()) * 1.3
print(f'separators: {seps:,}')
print(f'approx tokens: {tokens_approx:,.0f}')
print(f'separators per 1000 tokens: {1000 * seps / tokens_approx:.3f}')
"
```

Compare the printed "separators per 1000 tokens" figure against the currently-shipped
blend's own figure (798,771 separators / 399,508,203 tokens ≈ 2.00 per 1000, per this
project's own committed measurements). Per the spec's Gate 1, this must land within ±25% of
that figure (i.e. roughly 1.5–2.5 per 1000). If it falls outside that band, the fix is
revisiting `pg19`'s `words_per_document` value in `train/corpus.py` (larger value = fewer,
sparser separators; smaller = more, denser) and re-running Tasks 8–10's fetch/prepare/blend
steps — this is exactly the kind of pre-training check this project's history says pays for
itself, and it is deliberately a manual measurement here rather than a new automated test,
since it depends on the real, freshly-built blend rather than on code behaviour.

- [ ] **Step 3: Gate 2 — retrain the tokenizer at 48K**

Run: `python scripts/build_tokenizer.py --vocab-size 48000`
Expected: `>> Using existing corpus at artifacts/corpus/blend.txt ...`, then
`>> Training 48000-token BPE -> artifacts/tokenizer`, then
`>> Achieved vocabulary: 48,000 (target: 48,000)`. If the achieved vocabulary falls short,
this is `scripts/build_tokenizer.py`'s own documented hard-fail path (Review Focus item 5) —
confirm it actually exits non-zero rather than silently continuing, and if it fires, the
blend genuinely needs more distinct text, not a lower `--vocab-size`.

- [ ] **Step 4: Confirm the retrain didn't silently break shape guarantees**

Run: `python -m pytest tests/test_corpus.py tests/test_sizes.py -q`
Expected: PASS — `tests/test_sizes.py` doesn't depend on the tokenizer's contents, only on
the registry/YAML pair, but running it here catches an accidental edit to
`train/configs/model/tt-tnt-1024v48k.yaml` during this task before it goes any further
undetected.

- [ ] **Step 5: Commit the tracked manifest**

```bash
git add docs/measurements/blend_manifest.json
git commit -m "data: final blend + 48K tokenizer built against the settled corpus"
```

---

## Task 11: Tokenize the final blend

**Files:**
- No new source files; produces `artifacts/tokens-storyreg/{train,val}_ids.npy`.

**Interfaces:**
- Consumes: `artifacts/corpus/blend.txt`, `artifacts/tokenizer` (Task 10),
  `train/tokenization.py`.
- Produces: the token arrays the follow-up training-run plan trains against.

- [ ] **Step 1: Tokenize with the stratified split**

Run:
```bash
python train/tokenization.py \
    --corpus artifacts/corpus/blend.txt \
    --tokenizer artifacts/tokenizer \
    --out artifacts/tokens-storyreg \
    --blend-manifest docs/measurements/blend_manifest.json
```

Expected: `artifacts/tokens-storyreg/train_ids.npy` and `.../val_ids.npy` written, with a
printed `TokenStats` summary (total/train/val token counts, vocab size 48,000). The
`--blend-manifest` flag drives the STRATIFIED split (an even validation share held out from
each source's own span) rather than the default whole-stream tail split — this project's own
established reason: a tail-only split on a multi-source corpus risks a validation set drawn
almost entirely from whichever source was written last into the blend.

- [ ] **Step 2: Sanity-check the token stream against the model's vocabulary**

Run:
```bash
python3 -c "
import numpy as np
train = np.load('artifacts/tokens-storyreg/train_ids.npy')
val = np.load('artifacts/tokens-storyreg/val_ids.npy')
print('train max id:', int(train.max()), 'val max id:', int(val.max()))
assert int(train.max()) < 48000
assert int(val.max()) < 48000
print('OK: both splits stay within the 48,000-token vocabulary')
"
```

Expected: `OK: both splits stay within the 48,000-token vocabulary`. This exercises exactly
the check Task 1 fixed in `train/run.py` (the follow-up training-run plan will hit the real
guarded path; this step confirms the data itself is consistent before that plan ever runs).

- [ ] **Step 3: Confirm the full CPU-only suite is still green**

Run: `python -m pytest -q`
Expected: PASS (any pre-existing, unrelated skips are fine and expected, matching this
project's baseline — no new failures).

- [ ] **Step 4: Commit**

Nothing under `artifacts/` is committed (gitignored). If any tracked file changed as a side
effect (none expected), commit it; otherwise this step is a no-op confirmation via
`git status --short` showing no unexpected tracked changes.

---

## What this plan does not do

It does not run the actual multi-seed training run or Gates 4–6 (register comparison against
the current `tt-tnt-1024` checkpoint, guardrail checks, the qualitative read on prompt set C)
— that is a separate follow-up plan, once this one's corpus and tokenizer exist, matching
this project's own precedent (`2026-08-13-corpus-assembly.md` vs.
`2026-08-12-real-training-run.md` were separate plans for an analogous prior stage). It does
not touch context length, capacity, or any fine-tuning objective. It does not publish
anything to the Hub or change `docs/current_model.json`'s designation.
