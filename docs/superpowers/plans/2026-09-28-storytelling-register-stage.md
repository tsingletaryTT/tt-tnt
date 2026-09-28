<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Storytelling Register Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train a new pretraining line (fresh init, one unified blend) that adds `deepmind/pg19` (large, license-clean, lexically rich narrative text) and a small hand-curated "tortoise" flavour slice to the existing corpus registry, retrains the tokenizer at 48K, and produces two seed replicates evaluated for register/voice against the current designated `tt-tnt-1024` checkpoint.

**Architecture:** Everything routes through the existing corpus pipeline (`train/corpus.py` registry → `fetch_corpus.py` → `prepare_corpus.py` → `measure_corpus.py` → `blend_corpus.py` → `build_tokenizer.py` → `train/tokenization.py` → `train/run.py` → `convert/to_hf.py` → `scripts/evaluate.py`), extended in place rather than forked: PG19 needs a new fetch-time chunking mechanism (its rows are whole novels), Tortoise needs a new "local" (not HF-fetched) source kind, and the tokenizer/model registry need to stop assuming there will only ever be one 32K vocabulary.

**Tech Stack:** Python 3.10+, `datasets`, `tokenizers`, `transformers`, `numpy`, `pytest`. Training tasks need one leased Blackhole chip (or four via `--ddp 4`) and `ttml`; every other task is CPU-only.

**Spec:** [`docs/superpowers/specs/2026-09-28-storytelling-register-stage-design.md`](../specs/2026-09-28-storytelling-register-stage-design.md)

## Global Constraints

- SPDX header pair (`# SPDX-License-Identifier: Apache-2.0` / `# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC`) on every new file. Python 3.10+.
- No bare `assert` for guards in production code (tests may assert freely).
- `pyproject.toml` must NOT be modified; no new dependencies — PG19 fetches through the same `datasets.load_dataset(..., streaming=True)` path already used for FineWeb-Edu/Gutenberg/Wikipedia.
- **DELETE NOTHING under the current published baseline**: `artifacts/checkpoints-stageb/`, `artifacts/hf-tt-tnt-1024/`, `artifacts/tokens-v3/`, `artifacts/tokens-v4/`, `artifacts/tokenizer/`. This entire plan writes to NEW paths only (`artifacts/tokenizer-storyreg/`, `artifacts/tokens-storyreg/`, `artifacts/checkpoints-storyreg-<seed>/`, `artifacts/hf-storyreg-<seed>/`). **Never write to the shared `artifacts/tokenizer/` directory** — every existing checkpoint header still names that exact path for provenance, and overwriting it in place would silently invalidate that provenance for the published model.
- Run `python scripts/check_disk_space.py` (or inspect `shutil.disk_usage`) before any bulk fetch or blend write; this machine has repeatedly run near capacity (see CLAUDE.md's 2026-08-31 prune and 2026-09-01 entries).
- `train/corpus.py`'s `target_share` values must still sum to exactly 1.0 (`tests/test_corpus.py::test_target_shares_sum_to_one`).
- Licensing is DATA, not prose — lives in `CorpusSource` fields and is rendered via `scripts/render_licensing.py`, never hand-written into the README.
- The corpus is never redistributed; `artifacts/` stays gitignored. Ship a recipe (registry + scripts), not the corpus text.
- No `import ttml`/`import ttnn` without an active `gozer` lease. This applies to Task 10 (the only device-touching task) — use `importlib.util.find_spec` for any existence check elsewhere.
- Never overwrite an existing `train_ids.npy`/`val_ids.npy`. `tokenize_corpus` already refuses unless `overwrite=True`/`--force` — always target a NEW `--out` directory instead of passing that flag.
- `docs/current_model.json` is **not** touched by this plan. Designation is a deliberate, separate, human-reviewed decision made after Task 12's results exist — never a side effect of a training run or a script.

## Review Focus

- **A PG19 chunk boundary that respects paragraph breaks can still land at an awkward narrative moment** (mid-scene, mid-dialogue exchange) even though no sentence or paragraph is literally split. Task 2's tests check word counts and paragraph integrity; a human read of a handful of real chunk boundaries is the check that actually catches "technically valid, still jarring."
- **A `fetch_kind="local"` source has no `hf_repo`/`hf_revision`, and the existing fetch-spec test only branches on `"url"` vs. the implicit `"hf"` default.** Task 3 must update that branching explicitly, not just add the source and hope the existing test's `else` branch happens to tolerate it.
- **Raising the upsample cap for the flavour-class sources must be scoped to exactly those sources**, not implemented as a single global default that a later, unrelated source could also silently exploit. Task 6's mechanism and its test must name the seven sources explicitly.
- **`train.config.VOCAB_SIZE` still exists after Task 1's fix, kept as a legacy default for old call sites and tests.** A reviewer should confirm no remaining code path treats it as authoritative for a NEW run — only `RunConfig`'s own resolved model config, read fresh each invocation, should decide.
- **Prompt set C's entries must be scene-openings a model can continue, not bare titles.** `"Djed"` alone gives a generative model nothing to extend; every entry needs a sentence fragment that puts the title's image in motion, checked in Task 4's tests by a minimum-word-count floor per prompt, not just a title string.

---

## File Structure

| File | Responsibility |
|---|---|
| `train/run.py` | Task 1: `--tokenizer-dir` flag; stops trusting the global `VOCAB_SIZE` constant. |
| `train/checkpoint.py` | Task 1: `build_header` gains a `vocab_size` parameter. |
| `scripts/fetch_corpus.py` | Task 2: paragraph-aligned book chunking for PG19. Task 3: `fetch_kind="local"` support. |
| `train/corpus.py` | Task 2: `pg19` entry. Task 3: `tortoise` entry, `fetch_kind` allows `"local"`. Task 6: re-settled shares. |
| `train/tortoise_vignettes.py` | Task 3: the committed hand-authored corpus (titles + vignettes). |
| `docs/evaluation_prompts_c.json` | Task 4: the third frozen prompt set. |
| `tests/test_evaluation_prompts_c.py` | Task 4: its schema/digest tests. |
| `scripts/score_behaviour.py` | Task 4: registers prompt set `"c"`. |
| `scripts/check_separator_density.py` | Task 5: the new Gate 1 tool. |
| `train/sizes.py`, `train/configs/model/tt-tnt-1024.yaml` | Task 8: `vocab_size` 32000 → 48000. |
| `docs/corpus_blend.md` | Task 12: describes the new blend, with the same "not what's published" banner the current one carries. |

---

## Task 1: `train/run.py` can train against a tokenizer that isn't the shared default

**Files:**
- Modify: `train/checkpoint.py` (`build_header`, ~line 71-135)
- Modify: `train/run.py` (`main()`, the `build_yaml_config` call ~line 897 and the `build_header` call ~line 1142; the vocab-mismatch guard ~line 991-1011; a new `--tokenizer-dir` CLI flag)
- Test: `tests/test_checkpoint.py` (extend), `tests/test_training_config.py` or equivalent existing run.py CLI test file (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_header(..., vocab_size: int = VOCAB_SIZE, ...)` — callers may now pass the real vocabulary a run trained against; `train/run.py --tokenizer-dir PATH` (default: `artifacts/tokenizer`, unchanged for every existing invocation).

**Why this is a prerequisite.** This project has had exactly one tokenizer for its entire history, so two real bugs have never been exercised: (1) `train/run.py`'s vocab-mismatch guard compares the model config's declared `vocab_size` against `train.config.VOCAB_SIZE` — a hardcoded module constant — instead of against the tokenizer actually in use, so the check is (per its own inline comment) "asserting its own constant against itself"; (2) `build_header` always stamps a checkpoint's `vocab_size` field from that same constant, and `train/run.py` always records `tokenizer_dir` as the literal path `artifacts/tokenizer`, regardless of which tokenizer directory the token arrays were actually encoded with. Retraining the tokenizer (Task 8) makes both of these live bugs: every checkpoint from this line would carry a false provenance record pointing at the wrong tokenizer and the old vocab size. Fixed once, here, before anything downstream depends on it.

- [ ] **Step 1: Write the failing test for `build_header`'s new parameter**

```python
# add to tests/test_checkpoint.py
def test_build_header_records_the_vocab_size_it_is_given_not_the_module_default():
    """vocab_size must describe the model that produced these weights, the same way
    seq_len already does (see build_header's own docstring on seq_len) -- a header
    that silently records train.config.VOCAB_SIZE regardless of the caller's real
    tokenizer is a provenance lie the moment a second tokenizer exists."""
    header = build_header(
        1, model_config_path="m.yaml", tokenizer_dir="t", corpus_tokens=100,
        batch_size=8, seed=1, tokens_dir="tok", optimizer={}, ddp=1,
        vocab_size=48000,
    )
    assert header["vocab_size"] == 48000


def test_build_header_vocab_size_defaults_to_the_module_constant():
    """Existing callers (old tests, ad-hoc scripts) that don't pass vocab_size keep
    getting today's behaviour exactly -- this is what makes the change additive."""
    header = build_header(
        1, model_config_path="m.yaml", tokenizer_dir="t", corpus_tokens=100,
        batch_size=8, seed=1, tokens_dir="tok", optimizer={}, ddp=1,
    )
    from train.config import VOCAB_SIZE
    assert header["vocab_size"] == VOCAB_SIZE
```

- [ ] **Step 2: Run to verify the first test fails**

Run: `python -m pytest tests/test_checkpoint.py -k vocab_size -q`
Expected: FAIL — `build_header() got an unexpected keyword argument 'vocab_size'`

- [ ] **Step 3: Add the parameter**

In `train/checkpoint.py`, change `build_header`'s signature (currently ending `seq_len: int = SEQ_LEN, extra: Optional[Dict[str, Any]] = None,`) to:

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
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_SIZE,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
```

And in the body, change the header dict's `"vocab_size": VOCAB_SIZE,` line to `"vocab_size": int(vocab_size),`. Update the docstring's `vocab_size` paragraph (currently: "recorded from `train.config` rather than passed in: it must describe the model that produced these weights, and taking it from the single source of truth removes the chance of a caller recording something else") to instead read:

```
    ``vocab_size`` defaults to ``train.config.VOCAB_SIZE`` for callers that don't care
    (tests, ad-hoc scripts) but the real training call site (``train/run.py``) always
    passes it explicitly, taken from the model config actually in use for THIS run --
    exactly the same reasoning ``seq_len`` already documents two paragraphs up. A second
    tokenizer existing at all is what makes the module constant unsafe as a silent
    default for a real run: it is only ever "the one tokenizer this project has had until
    now", never a guarantee about the one a given run used.
```

- [ ] **Step 4: Run to verify both tests pass**

Run: `python -m pytest tests/test_checkpoint.py -k vocab_size -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Write the failing test for `--tokenizer-dir` and the fixed vocab check**

Find the existing test file that drives `train/run.py --dry-run` (grep `tests/` for `"--dry-run"` and `"build_yaml_config"` if the exact filename isn't obvious; extend whichever one already invokes `main()` with argv). Add:

```python
def test_dry_run_accepts_a_custom_tokenizer_dir(tmp_path, monkeypatch, capsys):
    """--tokenizer-dir must reach build_yaml_config's tokenizer_dir positional arg,
    not just be parsed and ignored."""
    import train.run as run_mod
    argv = ["--dry-run", "--tokenizer-dir", str(tmp_path / "custom-tok"),
            "--tokens-dir", str(tmp_path / "tok-ids")]
    # A --dry-run exits before opening a device or reading token arrays, so no fixture
    # files need to exist on disk for this to reach the config-assembly code under test.
    monkeypatch.setattr("sys.argv", ["run.py"] + argv)
    rc = run_mod.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert str(tmp_path / "custom-tok") in out


def test_vocab_mismatch_check_no_longer_reads_the_global_constant():
    """The guard must compare against the MODEL CONFIG'S declared vocab_size, not
    train.config.VOCAB_SIZE -- otherwise a model correctly configured for a NEW
    tokenizer (e.g. 48000) fails this check purely because the global constant still
    says 32000."""
    import inspect
    import train.run as run_mod
    src = inspect.getsource(run_mod.main)
    # This is a source-text check, deliberately: the bug this guards is a comparison
    # against the wrong THING, and a functional test would need a full CLI invocation
    # with two real vocab sizes to observe it, which duplicates test_dry_run_... above
    # for less signal. Confirms the exact defect is gone from the source, not just that
    # today's one call site happens not to trip it.
    assert "model_vocab_size != VOCAB_SIZE" not in src
    assert "train_ids.max()) >= VOCAB_SIZE" not in src
```

- [ ] **Step 6: Run to verify both fail**

Run: `python -m pytest tests/test_training_config.py -k "tokenizer_dir or vocab_mismatch" -q` (adjust the filename to whichever file you extended in Step 5)
Expected: FAIL — no `--tokenizer-dir` argument, and the old comparisons are still present.

- [ ] **Step 7: Implement in `train/run.py`**

Add the CLI flag near the other path flags (alongside `--tokens-dir`):

```python
    p.add_argument("--tokenizer-dir", default=str(ROOT / "artifacts" / "tokenizer"),
                   help="Directory holding the BPE tokenizer this run's token arrays "
                        "were encoded with (default: %(default)s, the shared default "
                        "every run before this flag existed used implicitly). Recorded "
                        "in the checkpoint header for provenance -- training itself "
                        "never loads the tokenizer object, since it operates on "
                        "pre-tokenized .npy id arrays.")
```

Change the `build_yaml_config` call site from:

```python
    yaml_config = build_yaml_config(
        str(ROOT / "artifacts" / "tokenizer"), str(model_config),
```

to:

```python
    yaml_config = build_yaml_config(
        args.tokenizer_dir, str(model_config),
```

Change the vocab-mismatch guard block from:

```python
    if model_vocab_size != VOCAB_SIZE:
        raise ValueError(
            f"model config declares vocab_size={model_vocab_size} but train.config.VOCAB_SIZE "
            f"is {VOCAB_SIZE}; the tokenizer and the model disagree"
        )
    if int(train_ids.max()) >= VOCAB_SIZE:
        raise ValueError(
            f"token id {int(train_ids.max())} exceeds vocab_size {VOCAB_SIZE}; these tokens "
            "were produced by a different tokenizer than the model config expects"
        )
```

to:

```python
    # THE REAL CHECK, using the model's OWN declared vocab_size rather than the module
    # constant train.config.VOCAB_SIZE -- which described "the one tokenizer this
    # project has had" and stopped being trustworthy the day a second tokenizer (a
    # different vocab size entirely) existed. The redundant self-comparison this
    # block used to do (model_vocab_size != VOCAB_SIZE, i.e. asserting the constant
    # against itself, per this function's own prior comment) is gone: there is
    # nothing left to compare model_vocab_size against except the data.
    if int(train_ids.max()) >= model_vocab_size:
        raise ValueError(
            f"token id {int(train_ids.max())} exceeds vocab_size {model_vocab_size}; "
            "these tokens were produced by a different tokenizer than the model "
            "config expects"
        )
```

Remove the now-unused `VOCAB_SIZE` import from `train/run.py`'s import block (grep the file for other uses first — there should be none left after this change).

Change the `build_header` call site (in `_save_checkpoint`) to add, alongside the existing explicit `seq_len=cfg.seq_len,`:

```python
                    tokenizer_dir=args.tokenizer_dir,
```

(replacing the hardcoded `tokenizer_dir=str(ROOT / "artifacts" / "tokenizer"),`), and add a new line:

```python
                    # Explicit, same reasoning as seq_len two lines up: a second
                    # tokenizer now exists, so the header must record what THIS run
                    # actually used, not train.config.VOCAB_SIZE's legacy default.
                    vocab_size=model_vocab_size,
```

- [ ] **Step 8: Run to verify all four new tests pass**

Run: `python -m pytest tests/test_checkpoint.py tests/test_training_config.py -k "vocab_size or tokenizer_dir or vocab_mismatch" -q`
Expected: PASS (4 tests)

- [ ] **Step 9: Run the full suite to confirm no existing test regressed**

Run: `python -m pytest -q`
Expected: same pass count as before this task, plus 4.

- [ ] **Step 10: Commit**

```bash
git add train/checkpoint.py train/run.py tests/test_checkpoint.py tests/test_training_config.py
git commit -m "feat(train): support a non-default tokenizer directory in run.py

Prerequisite for retraining the tokenizer (storytelling-register stage):
build_header now records the real vocab_size it is given instead of the
global train.config constant, and --tokenizer-dir lets a run point at a
tokenizer other than the shared artifacts/tokenizer default, so its
checkpoint header's provenance is never a silent lie."
```

---

## Task 2: Register `deepmind/pg19` with paragraph-aligned book chunking

**Files:**
- Modify: `scripts/fetch_corpus.py` (`chunk_by_paragraphs`, `CHUNKED_RENDERERS`, `iter_source_rows`)
- Modify: `train/corpus.py` (new `pg19` entry in `SOURCES`)
- Modify: `README.md` (provenance section — Gate 0 of the spec)
- Test: `tests/test_fetch_corpus.py` (extend), `tests/test_corpus.py` (extend — parametrised tests cover it automatically once it's in `SOURCES`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `chunk_by_paragraphs(text: str, min_words: int = 2000, max_words: int = 4000) -> List[str]`; `CHUNKED_RENDERERS: Dict[str, Callable[[dict], List[str]]]`, checked by `iter_source_rows` before `RENDERERS`/`TEXT_COLUMN`.

- [ ] **Step 1: The revision, schema, and streaming behaviour — already verified, re-verify before trusting it**

Verified directly against the real HF Hub API and a real streaming load during this plan's own writing (not guessed, and independently re-confirmed after a research fork's contradictory claim turned out to be wrong — see below):

- **Pinned revision:** `4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b` (from `HfApi().dataset_info('deepmind/pg19').sha`).
- **Streams with the exact same call every other source in this registry already uses** — `datasets.load_dataset('deepmind/pg19', split='train', streaming=True)` — **no `trust_remote_code` needed.** (A separate research pass had claimed PG19 was script-based and would require `trust_remote_code=True`, proposing a bespoke per-URL fetch script instead; that claim did not reproduce under a direct test and is not used here. If it fails differently in your environment, that is a real, new finding worth reporting — re-run the exact one-line check below before assuming the bespoke-fetch route is needed.)
- **Real row schema:** `['publication_date', 'short_book_title', 'text', 'url']` — the book text is in `"text"`, matching this plan's `TEXT_COLUMN`/`CHUNKED_RENDERERS` assumption exactly; `short_book_title` is available for later human inspection (e.g. Task 7 Step 3's chunk-boundary read) but not required for training.

Re-run this exact check before proceeding, to confirm nothing has changed since this plan was written:
```bash
python -c "
from huggingface_hub import HfApi
from datasets import load_dataset
info = HfApi().dataset_info('deepmind/pg19')
assert info.sha == '4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b', f'revision moved: {info.sha}'
ds = load_dataset('deepmind/pg19', split='train', streaming=True)
row = next(iter(ds))
assert sorted(row.keys()) == ['publication_date', 'short_book_title', 'text', 'url'], row.keys()
print('confirmed: revision, schema, and streaming all match this plan')
"
```
If this fails for any reason, STOP and report the real error rather than substituting a guess — `CorpusSource.__post_init__` already refuses an empty `hf_revision`, and the point of pinning is that this exact value is what gets fetched, not an approximation.

- [ ] **Step 2: Write the failing tests for `chunk_by_paragraphs`**

```python
# add to tests/test_fetch_corpus.py
from scripts.fetch_corpus import chunk_by_paragraphs


def test_chunk_by_paragraphs_splits_a_long_text_into_multiple_chunks():
    paragraphs = [f"Paragraph {i} has some words in it, several of them." for i in range(400)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_by_paragraphs(text, min_words=200, max_words=400)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert len(chunk.split()) >= 200 or chunk is chunks[-1]


def test_chunk_by_paragraphs_never_splits_a_paragraph_across_chunks():
    paragraphs = [f"Paragraph number {i}, unique marker XYZ{i}." for i in range(50)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_by_paragraphs(text, min_words=50, max_words=100)
    reconstructed = "\n\n".join(chunks)
    # Every original paragraph string appears intact in exactly one chunk -- proof no
    # paragraph was cut mid-sentence to hit a word-count target.
    for para in paragraphs:
        assert para in reconstructed
        assert sum(para in c for c in chunks) == 1


def test_chunk_by_paragraphs_final_chunk_may_be_shorter_than_min_words():
    text = "\n\n".join(f"Paragraph {i} text here." for i in range(3))
    chunks = chunk_by_paragraphs(text, min_words=10_000, max_words=20_000)
    # Nothing in this text reaches even min_words once, so it must all land in one
    # chunk rather than being dropped or raising.
    assert len(chunks) == 1


def test_chunk_by_paragraphs_a_single_oversized_paragraph_becomes_its_own_chunk():
    """A paragraph longer than max_words is never split mid-paragraph -- it becomes one
    chunk larger than the target, which is the honest tradeoff this function makes."""
    huge_paragraph = " ".join(f"word{i}" for i in range(5000))
    text = "short first paragraph.\n\n" + huge_paragraph + "\n\nshort last paragraph."
    chunks = chunk_by_paragraphs(text, min_words=100, max_words=1000)
    assert any(len(c.split()) > 1000 for c in chunks)


def test_chunk_by_paragraphs_empty_text_returns_no_chunks():
    assert chunk_by_paragraphs("", min_words=100, max_words=200) == []
    assert chunk_by_paragraphs("   \n\n  ", min_words=100, max_words=200) == []


def test_chunk_by_paragraphs_no_blank_lines_at_all_returns_one_chunk_regardless_of_size():
    """A book with no double-newlines (unusual, but PG19's OCR-derived text can have
    ragged formatting) must not crash or silently drop content."""
    text = "one very long line with no paragraph breaks whatsoever and many words " * 50
    chunks = chunk_by_paragraphs(text, min_words=10, max_words=50)
    assert len(chunks) == 1
    assert chunks[0].strip() == text.strip()
```

- [ ] **Step 3: Run to verify they fail**

Run: `python -m pytest tests/test_fetch_corpus.py -k chunk_by_paragraphs -q`
Expected: FAIL — `ImportError: cannot import name 'chunk_by_paragraphs'`

- [ ] **Step 4: Implement `chunk_by_paragraphs` in `scripts/fetch_corpus.py`**

```python
def chunk_by_paragraphs(text: str, min_words: int = 2000, max_words: int = 4000) -> List[str]:
    """Split ``text`` into paragraph-aligned chunks, each within [min_words, max_words].

    Built for PG19: its rows are whole novels (tens of thousands of words), and treating
    one book as one document would make the end-of-document separator so rare per token
    that termination behaviour degrades the same way the TinyStories-share reduction
    already measured (see this project's 2026-08-27 CLAUDE.md entry: cutting TinyStories'
    share thinned </s> density and cost termination rate). Chunking restores a sane
    separator frequency without ever splitting a paragraph mid-sentence.

    A paragraph is delimited by a blank line (``\\n\\n``), matching the convention every
    other tool in this pipeline already uses (``scripts/measure_corpus.py``,
    ``scripts/blend_corpus.py``'s ``TokenMeter``). A chunk accumulates whole paragraphs
    until adding the next one would exceed ``max_words`` AND the chunk has already
    reached ``min_words`` -- so a chunk under ``min_words`` always accepts one more
    paragraph rather than closing early, and a single paragraph larger than ``max_words``
    becomes its own oversized chunk rather than being split. The final chunk may be
    shorter than ``min_words``: it is whatever is left over, not padded or dropped.
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0
    for para in paragraphs:
        para_words = len(para.split())
        if current and current_words >= min_words and current_words + para_words > max_words:
            chunks.append("\n\n".join(current))
            current = []
            current_words = 0
        current.append(para)
        current_words += para_words
    if current:
        chunks.append("\n\n".join(current))
    return chunks
```

- [ ] **Step 5: Run to verify all six tests pass**

Run: `python -m pytest tests/test_fetch_corpus.py -k chunk_by_paragraphs -q`
Expected: PASS (6 tests)

- [ ] **Step 6: Write the failing test for the chunked-renderer path in `iter_source_rows`**

```python
# add to tests/test_fetch_corpus.py
from train.corpus import CorpusSource
from scripts.fetch_corpus import iter_source_rows, CHUNKED_RENDERERS


def test_chunked_renderer_yields_one_row_per_chunk(monkeypatch):
    """A CHUNKED_RENDERERS entry must expand ONE upstream row into MULTIPLE {"text": ...}
    rows -- the mechanism PG19 needs, generalised so any future one-row-per-book source
    can reuse it without touching iter_source_rows again."""
    def fake_chunker(row):
        return [f"chunk-{i}-of-{row['text']}" for i in range(3)]

    monkeypatch.setitem(CHUNKED_RENDERERS, "some/repo", fake_chunker)

    class FakeRow(dict):
        pass

    def fake_load_dataset(repo, **kwargs):
        return [FakeRow(text="book one")]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    src = CorpusSource(name="fake", slice="spine", target_share=0.0,
                       hf_repo="some/repo", hf_revision="a" * 40)
    rows = list(iter_source_rows(src))
    assert [r["text"] for r in rows] == [
        "chunk-0-of-book one", "chunk-1-of-book one", "chunk-2-of-book one",
    ]
```

- [ ] **Step 7: Run to verify it fails**

Run: `python -m pytest tests/test_fetch_corpus.py -k chunked_renderer -q`
Expected: FAIL — `ImportError: cannot import name 'CHUNKED_RENDERERS'`

- [ ] **Step 8: Implement the chunked-renderer path**

In `scripts/fetch_corpus.py`, add near `RENDERERS`:

```python
#: Sources whose upstream row is LARGER than a document, the inverse of `poetry`'s
#: `rows_per_document` (where a row is smaller than a document). A chunking renderer
#: takes one row and returns a LIST of document strings rather than one string --
#: PG19's rows are whole novels, so each row becomes several ~2000-4000-word,
#: paragraph-aligned documents. Checked before RENDERERS/TEXT_COLUMN in
#: iter_source_rows, so a source may use either mechanism but not both.
def _render_pg19_chunks(row: Dict[str, object]) -> List[str]:
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return []
    return chunk_by_paragraphs(text, min_words=2000, max_words=4000)


CHUNKED_RENDERERS: Dict[str, "Callable[[Dict[str, object]], List[str]]"] = {
    "deepmind/pg19": _render_pg19_chunks,
}
```

(Add `Callable` and `List` to the existing `from typing import ...` line if not already imported.)

Modify `iter_source_rows` to check `CHUNKED_RENDERERS` first:

```python
def iter_source_rows(source: CorpusSource, limit_rows: int = 0) -> Iterator[Dict[str, object]]:
    """..."""  # existing docstring unchanged
    if source.fetch_kind == "url":
        yield from _iter_url_rows(source, limit_rows)
        return

    from datasets import load_dataset

    chunker = CHUNKED_RENDERERS.get(source.hf_repo)
    renderer = RENDERERS.get(source.hf_repo)
    column = TEXT_COLUMN.get(source.hf_repo)
    if chunker is None and renderer is None and column is None:
        raise ValueError(
            f"no text column, renderer, or chunker registered for {source.hf_repo}; "
            f"add it to TEXT_COLUMN, RENDERERS, or CHUNKED_RENDERERS"
        )

    kwargs = {"split": source.hf_split, "revision": source.hf_revision, "streaming": True}
    if source.hf_config:
        kwargs["name"] = source.hf_config
    ds = load_dataset(source.hf_repo, **kwargs)

    seen = 0
    for row in ds:
        seen += 1
        if limit_rows and seen > limit_rows:
            return

        if source.hf_repo == GUTENBERG_REPO:
            md = row.get("METADATA")
            if isinstance(md, str):
                try:
                    md = json.loads(md)
                except json.JSONDecodeError:
                    continue
            if not isinstance(md, dict) or not matches_source(md, source):
                continue

        if chunker is not None:
            for chunk_text in chunker(row):
                if chunk_text.strip():
                    yield {"text": chunk_text}
            continue

        text = renderer(row) if renderer is not None else row.get(column)
        if not isinstance(text, str) or not text.strip():
            continue
        yield {"text": text}
```

- [ ] **Step 9: Run to verify it passes**

Run: `python -m pytest tests/test_fetch_corpus.py -k chunked_renderer -q`
Expected: PASS

- [ ] **Step 10: Register `pg19` in `train/corpus.py`**

Add to `SOURCES`, using the revision confirmed in Step 1:

```python
    "pg19": CorpusSource(
        name="pg19",
        slice="spine",
        # Target share settled in Task 6 against real measured availability; 0.0 here
        # is a deliberate placeholder ONLY until that task runs -- see its own
        # docstring note in measure_corpus.py: shares are targets revised on evidence,
        # never guessed in advance.
        target_share=0.0,
        hf_repo="deepmind/pg19",
        hf_revision="4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b",
        hf_split="train",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/datasets/deepmind/pg19",
        attribution="PG-19 Language Modelling Benchmark (Rae et al., DeepMind), "
                    "deepmind/pg19",
        license_note=(
            "Apache-2.0 covers DeepMind's packaging of this dataset. The underlying "
            "texts are pre-1919 novels, independently public domain by age -- the "
            "stronger and more durable claim, matching how sedthh/gutenberg_english's "
            "MIT packaging is distinguished from its own public-domain texts elsewhere "
            "in this registry."
        ),
        rows_per_document=1,
        rationale=(
            "Large-scale, lexically rich narrative fiction: full novels, not the small "
            "hand-curated slices (spine/folklore/weird/flavour) this registry already "
            "carries, and not TinyStories' ~13,800-word vocabulary ceiling (measured in "
            "the embedding-geography and reach-dial work). Each upstream row is one "
            "whole book; scripts/fetch_corpus.py's CHUNKED_RENDERERS splits it into "
            "~2000-4000-word paragraph-aligned sub-documents before this pipeline ever "
            "sees it as a 'document', so document-separator density stays sane instead "
            "of firing once per novel. NOTE: DeepMind's own dataset card flags "
            "19th-century diction/social attitudes -- watch for archaic-voice drift, "
            "not just vocabulary richness, in the eventual eval (Gate 6 of the design "
            "spec)."
        ),
    ),
```

- [ ] **Step 11: Update the README provenance section (Gate 0)**

Run `python scripts/render_licensing.py` to regenerate `docs/corpus_licensing.md` from the registry (confirm it now lists `pg19` with the Apache-2.0/public-domain note above). Then add a short paragraph to `README.md`'s existing corpus-provenance section (find it by searching for the existing TinyStories/Wikipedia licensing discussion) stating explicitly: PG19 is Apache-2.0-packaged, underlying texts public domain by age, and that this is a NEW source not yet part of any published model's training corpus — matching the honesty pattern `docs/corpus_blend.md`'s existing banner already uses for the long-context-corpus experiment.

- [ ] **Step 12: Run the full suite**

Run: `python -m pytest -q`
Expected: all existing tests pass; the parametrised registry tests in `tests/test_corpus.py` (e.g. `test_every_source_declares_a_licence`, `test_every_source_has_a_known_slice`) now also run against `pg19` and pass, since it declares a real `license_id`/`license_url` and a slice already in `SLICES`.

- [ ] **Step 13: Commit**

```bash
git add scripts/fetch_corpus.py train/corpus.py README.md tests/test_fetch_corpus.py
git commit -m "feat(corpus): register deepmind/pg19 with paragraph-aligned book chunking

PG19 is Apache-2.0-packaged, public-domain-by-age narrative fiction at
~2B tokens -- large-scale, lexically rich, cleanly licensed. Its rows are
whole novels, so a new CHUNKED_RENDERERS mechanism in fetch_corpus.py
splits each book into ~2000-4000-word paragraph-aligned sub-documents
before prepare_corpus.py ever sees them, keeping document-separator
density sane (the same mechanism that broke when TinyStories' share was
cut, applied in advance rather than discovered after training)."
```

---

## Task 3: The Tortoise corpus slice — hand-authored, committed, and honestly small

**Files:**
- Create: `train/tortoise_vignettes.py`
- Modify: `scripts/fetch_corpus.py` (`_iter_local_rows`, wired into `iter_source_rows`)
- Modify: `train/corpus.py` (`CorpusSource.__post_init__` allows `"local"`; new `tortoise` entry)
- Modify: `tests/test_corpus.py` (`test_every_source_has_a_resolvable_fetch_spec` gains a `"local"` branch)
- Test: `tests/test_tortoise_vignettes.py`, extend `tests/test_fetch_corpus.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `train.tortoise_vignettes.VIGNETTES: Tuple[Tuple[str, str], ...]` — `(title, vignette_text)` pairs; `fetch_kind="local"` on `CorpusSource`, using `source_url` re-purposed to hold a dotted Python import path (e.g. `"train.tortoise_vignettes"`) rather than a URL.

This source is original content written for this project — Tortoise's song/album titles are short phrases (not independently copyrightable), and the vignettes are new prose, not a redistribution of anything licensed. It is committed directly (unlike the HF sources, which ship a recipe rather than the text itself) because it is small, wholly owned, and there is no redistribution question to hedge.

- [ ] **Step 1: Write the failing tests for the vignette module's shape**

```python
# tests/test_tortoise_vignettes.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The hand-authored Tortoise corpus slice: real titles, original vignettes."""
from train.tortoise_vignettes import VIGNETTES
from scripts.score_behaviour import collapse_markers_found

#: Verified against Tortoise's real studio discography during this plan's brainstorming
#: (Wikipedia/Discogs/AllMusic cross-checked) -- not invented. Two Catastrophist tracks
#: with vocals ("Rock On", "Yonder Blue") are deliberately excluded from this
#: instrumental-titles slice.
EXPECTED_TITLES = {
    "A Simple Way to Go Faster Than Light That Does Not Work",
    "Djed", "Prepare Your Coffin", "Gigantes", "Seneca", "Eros", "Benway", "Gesceap",
    "The Suspension Bridge at Iguazu Falls", "Northern Something", "De Chelly",
    "Gopher Island", "His Second Story Island", "The Equator", "Everglade",
    "Ten-Day Interval", "Four-Day Interval", "The Fall of Seven Diamonds Plus One",
    "Monument Six One Thousand", "Almost Always Is Nearly Enough",
    "High Class Slim Came Floatin' In", "Shake Hands With Danger",
    "Swung from the Gutters", "I Set My Face to the Hillside", "At Odds With Logic",
    "Onions Wrapped in Rubber", "Tin Cans and Twine", "Salt The Skies",
    "The Lithium Stiffs", "Spiderwebbed", "Glass Museum", "Tesseract",
}


def test_every_expected_title_has_a_vignette():
    titles = {t for t, _ in VIGNETTES}
    missing = EXPECTED_TITLES - titles
    assert not missing, f"missing vignettes for: {sorted(missing)}"


def test_no_unexpected_titles_slipped_in():
    """A title not on the verified list is either a typo or an uncredited invention --
    both are the class of error this project explicitly guards against (see the global
    CLAUDE.md's 'check what a package is before installing it' lesson, same shape)."""
    titles = {t for t, _ in VIGNETTES}
    assert titles <= EXPECTED_TITLES, f"unexpected titles: {sorted(titles - EXPECTED_TITLES)}"


def test_no_title_is_duplicated():
    titles = [t for t, _ in VIGNETTES]
    assert len(titles) == len(set(titles))


def test_every_vignette_is_substantial():
    for title, text in VIGNETTES:
        word_count = len(text.split())
        assert word_count >= 150, f"{title!r}: only {word_count} words"


def test_no_vignette_trips_a_tinystories_collapse_marker():
    """The whole point of this slice is a register that ISN'T TinyStories' -- a vignette
    that opens 'Once upon a time' would be self-defeating."""
    for title, text in VIGNETTES:
        found = collapse_markers_found(text)
        assert not found, f"{title!r} contains collapse markers: {found}"


def test_vignette_text_is_not_a_bare_repetition_of_its_title():
    for title, text in VIGNETTES:
        assert text.strip().lower() != title.strip().lower()
        assert len(text) > len(title) * 5
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_tortoise_vignettes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'train.tortoise_vignettes'`

- [ ] **Step 3: Write the module — three worked examples plus the rest, same pattern**

Create `train/tortoise_vignettes.py`. The register throughout: observational-deadpan or found-document, in the same family as this project's existing `spine`/`weird` sources (Fabre, Fort, Machen) — never fairy-tale, never a moral, never "once upon a time." Three full examples (write the remaining 29 titles from `EXPECTED_TITLES` above in this same register, each 150-300 words, each avoiding every marker in `scripts/score_behaviour.py`'s `COLLAPSE_MARKERS`):

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Original vignettes seeded from real Tortoise (the band) song and album titles.

Titles are verified against the band's actual studio discography (Tortoise 1994,
Millions Now Living Will Never Die 1996, TNT 1998, Standards 2001, It's All Around You
2004, Beacons of Ancestorship 2009, The Catastrophist 2016) -- cross-checked against
Wikipedia, Discogs and AllMusic during this project's own brainstorming, not invented.
Short phrases are not independently copyrightable; every vignette below is new prose
written for this project, not a reproduction of anything licensed.

Titles are stored ASCII-normalised (e.g. "Iguazu" not "Iguazu with an acute accent on
the u", "and" not "&") for encoding safety in Python source and downstream text files --
a deliberate normalisation, not a data-fidelity error. The real, accented title is
"The Suspension Bridge at Iguazú Falls"; this module's "Iguazu" is the same title.

This is deliberately a SMALL, hand-curated, upsampled-and-capped source (see its
train/corpus.py rationale) -- a flavour injection akin to the existing 'flavour' slice
(Stein, the I Ching), not a bulk source. The register is observational-deadpan or
found-document, matching this project's existing 'spine'/'weird' voice (Fabre, Fort,
Machen), and deliberately clear of every marker in scripts/score_behaviour.py's
COLLAPSE_MARKERS -- this slice exists to counterweight, not reproduce, the TinyStories
register.
"""
from __future__ import annotations

from typing import Tuple

VIGNETTES: Tuple[Tuple[str, str], ...] = (
    (
        "A Simple Way to Go Faster Than Light That Does Not Work",
        "The apparatus took up most of the shed: a brass ring wound with four miles of "
        "copper wire, a bicycle dynamo repurposed from a threshing machine, and a glass "
        "sphere pumped down to something the inventor's notebook called 'as close to "
        "nothing as the shed allows'. The theory, set out over eleven pages in a hand "
        "that grew smaller and more certain as the pages went on, held that light could "
        "be persuaded to arrive before it left, given sufficient encouragement and a "
        "coil of the correct pitch. On the fourth of the month the dynamo was engaged "
        "for the span of one held breath. The sphere did not glow. The wire did not "
        "warm. A moth that had been resting on the ring's inner curve was found "
        "afterward on the outer curve, which the notebook records as 'inconclusive, "
        "possibly a draft.' Every subsequent attempt is logged with the same word: "
        "inconclusive. The shed still stands. The dynamo still turns, on request, for "
        "visitors who ask nicely and do not mind that nothing arrives sooner than it "
        "left. The notebook's final entry, undated, reads only: 'The wire hums at a "
        "frequency I have started to enjoy. This may be the actual result.'"
    ),
    (
        "Djed",
        "The pillar is described in four separate catalogues, and no two agree on its "
        "height. The first calls it a backbone, plainly, the vertebrae of a god rendered "
        "in stacked limestone discs, each disc slightly narrower than the one below it "
        "so that the whole column reads, from a distance, as something that could still "
        "stand up and walk away. The second catalogue calls it a ladder. The third, "
        "compiled two centuries later by a man who had only ever seen a drawing of a "
        "drawing, calls it a bundle of reeds tied at four points and never once "
        "mentions a spine at all. The fourth catalogue, the newest, simply measures: "
        "one meter nine, base to capital, discs numbering eleven, material consistent "
        "with a quarry eighty kilometers upriver. Every festival that raised one of "
        "these pillars did so at the turn of a season, hauling the column upright with "
        "ropes run through eyelets cut for exactly this purpose and no other, and the "
        "raising itself was the entire ceremony -- once it stood, the crowd dispersed, "
        "the priests packed their instruments, and the pillar was left to do whatever "
        "a raised backbone does once nobody is watching it anymore."
    ),
    (
        "Prepare Your Coffin",
        "The instructions arrived by the usual channel, folded into quarters and "
        "addressed in a hand the recipient did not recognize, which was itself not "
        "unusual: it had never once been the same hand twice. Measure twice at the "
        "shoulder, the letter said, and once at the widest point of the hip, and add "
        "two fingers' width to each figure for the wood's own thickness, which the "
        "carpenter will not think to mention unless asked directly. Choose a wood that "
        "answers when knocked -- pine answers thin, oak answers full, and a box that "
        "does not answer at all should be returned to whoever sold it. Line the "
        "interior only if you intend to be found by people who care about lining; "
        "otherwise the bare wood does its work regardless. The letter closed, as every "
        "letter in this series had closed, with the same six words, underlined once: "
        "this is not a threat. It was, the recipient had come to understand over the "
        "years, something closer to a maintenance schedule -- the kind of notice a "
        "well-run household sends itself about the roof, or the well, long before "
        "either one actually needs the attention."
    ),
    # The remaining 29 titles from EXPECTED_TITLES each get one vignette here, in this
    # exact register (observational-deadpan or found-document), 150-300 words, no
    # collapse markers. Titles remaining: Gigantes, Seneca, Eros, Benway, Gesceap, The
    # Suspension Bridge at Iguazu Falls, Northern Something, De Chelly, Gopher Island,
    # His Second Story Island, The Equator, Everglade, Ten-Day Interval, Four-Day
    # Interval, The Fall of Seven Diamonds Plus One, Monument Six One Thousand, Almost
    # Always Is Nearly Enough, High Class Slim Came Floatin' In, Shake Hands With
    # Danger, Swung from the Gutters, I Set My Face to the Hillside, At Odds With
    # Logic, Onions Wrapped in Rubber, Tin Cans and Twine, Salt The Skies, The Lithium
    # Stiffs, Spiderwebbed, Glass Museum, Tesseract.
)
```

- [ ] **Step 4: Run to verify all `test_tortoise_vignettes.py` tests pass**

Run: `python -m pytest tests/test_tortoise_vignettes.py -q`
Expected: PASS (6 tests) once all 32 titles have vignettes meeting the length/marker constraints above.

- [ ] **Step 5: Write the failing test for the `"local"` fetch kind**

```python
# add to tests/test_fetch_corpus.py
def test_local_fetch_kind_yields_the_vignette_rows():
    from train.corpus import CorpusSource
    from scripts.fetch_corpus import iter_source_rows

    src = CorpusSource(name="tortoise", slice="flavour", target_share=0.0,
                       hf_repo="", hf_revision="", fetch_kind="local",
                       source_url="train.tortoise_vignettes")
    rows = list(iter_source_rows(src))
    from train.tortoise_vignettes import VIGNETTES
    assert len(rows) == len(VIGNETTES)
    assert all(r["text"].strip() for r in rows)
```

Also add, to `tests/test_corpus.py`, the missing branch in the existing fetch-spec test (find `test_every_source_has_a_resolvable_fetch_spec` and add a third branch alongside the existing `if src.fetch_kind == "url":` one):

```python
def test_every_source_has_a_resolvable_fetch_spec(name):
    src = SOURCES[name]
    if src.fetch_kind == "url":
        assert src.source_url, f"{name}: fetch_kind='url' with no source_url"
        assert src.source_url.startswith("https://"), (
            f"{name}: source_url '{src.source_url}' is not https"
        )
        return
    if src.fetch_kind == "local":
        # source_url is re-purposed here to hold a dotted import path rather than a
        # URL -- the "fetch" is importing a committed Python module, not a network
        # request, so there is no revision to pin: the module IS the pinned content.
        assert src.source_url, f"{name}: fetch_kind='local' with no source_url"
        assert "://" not in src.source_url, (
            f"{name}: source_url '{src.source_url}' looks like a URL, not a "
            f"dotted import path"
        )
        import importlib
        module = importlib.import_module(src.source_url)
        assert hasattr(module, "VIGNETTES"), (
            f"{name}: {src.source_url} has no VIGNETTES attribute"
        )
        return
    assert src.hf_repo, f"{name}: no hf_repo"
```

(This replaces whatever the final `assert src.hf_repo` line currently is in that test — check the existing file for its exact tail before editing, since it may include additional lines after the `hf_repo` check.)

- [ ] **Step 6: Run to verify both fail**

Run: `python -m pytest tests/test_fetch_corpus.py -k local_fetch_kind -q tests/test_corpus.py -k resolvable_fetch_spec -q`
Expected: FAIL — `ValueError: fake: fetch_kind must be 'hf' or 'url'` (from `CorpusSource.__post_init__`) for the first; the second fails once `tortoise` is registered in Step 8 below and `"local"` isn't yet handled.

- [ ] **Step 7: Allow `"local"` in `CorpusSource.__post_init__`, and add `_iter_local_rows`**

In `train/corpus.py`, change:

```python
    def __post_init__(self) -> None:
        if self.fetch_kind not in ("hf", "url"):
```

to:

```python
    def __post_init__(self) -> None:
        if self.fetch_kind not in ("hf", "url", "local"):
```

and update the adjacent docstring comment on `fetch_kind` (currently: `#: How this source is fetched. "hf" is a HuggingFace dataset ... "url" is a direct download ...`) to add a third sentence: `#: "local" is committed, hand-authored content -- source_url holds a dotted Python import path to a module exposing a VIGNETTES tuple, not a URL.` Also update the `if self.fetch_kind == "url" and not self.source_url:` guard to cover `"local"` too, since both kinds need something in `source_url`:

```python
        if self.fetch_kind in ("url", "local") and not self.source_url:
            raise ValueError(f"{self.name}: fetch_kind={self.fetch_kind!r} needs a source_url")
```

In `scripts/fetch_corpus.py`, add near `_iter_url_rows`:

```python
def _iter_local_rows(source: CorpusSource) -> Iterator[Dict[str, object]]:
    """Rows from a committed Python module's ``VIGNETTES`` tuple.

    ``source.source_url`` holds a dotted import path (e.g. ``"train.tortoise_vignettes"``)
    rather than a URL for this fetch kind -- there is nothing to download, since the
    content is original prose committed directly to the repository rather than fetched
    from an external, licence-audited dataset.
    """
    import importlib

    module = importlib.import_module(source.source_url)
    for _title, text in module.VIGNETTES:
        if text.strip():
            yield {"text": text}
```

And in `iter_source_rows`, add the branch right after the existing `if source.fetch_kind == "url":` block:

```python
    if source.fetch_kind == "url":
        yield from _iter_url_rows(source, limit_rows)
        return
    if source.fetch_kind == "local":
        yield from _iter_local_rows(source)
        return
```

- [ ] **Step 8: Register `tortoise` in `train/corpus.py`**

```python
    "tortoise": CorpusSource(
        name="tortoise",
        slice="flavour",
        # Target share settled in Task 6, same as pg19 -- see measure_corpus.py's own
        # note that shares are targets revised on evidence. This source is tiny
        # (~25-35K raw tokens across ~32 vignettes) so its real ceiling, even at a
        # raised upsample cap, will be well under 1%.
        target_share=0.0,
        hf_repo="", hf_revision="",
        fetch_kind="local",
        source_url="train.tortoise_vignettes",
        license_id="",
        license_url="",
        attribution="Original vignettes written for this project, seeded from real "
                    "Tortoise (the band) song/album titles",
        license_note=(
            "No SPDX identifier applies: this is original prose written directly for "
            "this project, not a redistribution of anything licensed. Song/album "
            "TITLES are short phrases and not independently copyrightable; no lyrics "
            "or recorded material are used or reproduced anywhere in this source."
        ),
        # 12x, matching the raised cap Task 6 applies to every deliberately-small,
        # flavour-class source in this registry (spine/folklore/weird/flavour/
        # procedural/poetry/tortoise) for this run specifically -- see Task 6.
        upsample=12,
        rationale=(
            "This project's own name (tt-tnt) is understood, per the design spec, as "
            "Tenstorrent + Tortoise's 1998 album TNT -- confirmed, not invented: the "
            "band's real track 'A Simple Way to Go Faster Than Light That Does Not "
            "Work' is the origin of the faster-than-light canary prompt this project "
            "has run at every checkpoint since the qualitative-canary convention "
            "started. A small, honestly-scoped flavour injection (same class as "
            "'flavour': Stein, the I Ching), not a bulk source -- the real instrument "
            "for whether this pays off is the frozen eval prompt set C "
            "(docs/evaluation_prompts_c.json), not this corpus share, which is too "
            "small to move any aggregate metric on its own."
        ),
    ),
```

- [ ] **Step 9: Run to verify both new tests pass, and the full suite**

Run: `python -m pytest tests/test_fetch_corpus.py tests/test_corpus.py -q`
Expected: PASS, including the parametrised registry tests now also covering `tortoise`.

Run: `python -m pytest -q`
Expected: full suite green.

- [ ] **Step 10: Commit**

```bash
git add train/tortoise_vignettes.py scripts/fetch_corpus.py train/corpus.py \
        tests/test_tortoise_vignettes.py tests/test_fetch_corpus.py tests/test_corpus.py
git commit -m "feat(corpus): add the tortoise flavour slice (real titles, original vignettes)

New fetch_kind='local' for committed, hand-authored content that isn't
fetched from an external dataset. 32 vignettes seeded from Tortoise's
verified real discography, written in this project's existing
spine/weird register -- deliberately small and honestly scoped; the
real instrument for whether this pays off is eval prompt set C, not
this corpus share."
```

---

## Task 4: A third frozen evaluation prompt set, built from real Tortoise titles

**Files:**
- Create: `docs/evaluation_prompts_c.json`
- Create: `tests/test_evaluation_prompts_c.py`
- Modify: `scripts/score_behaviour.py` (`PROMPT_SETS` registers `"c"`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `PROMPT_SETS["c"]` (a `PromptSet` with `key="c"`, `suffix="-setC"`); `docs/evaluation_prompts_c.json` schema identical to sets A/B (`{"id", "probe", "text"}` per prompt).

Following set B's exact precedent: never pooled with A or B, its own `c-` id prefix, its own frozen digest. Every entry renders as a scene-opening (a sentence putting the title's image in motion), not a bare title — this is what makes it a prompt a generative model can meaningfully continue.

- [ ] **Step 1: Write the prompt file**

Create `docs/evaluation_prompts_c.json` using the same ~32 verified titles as Task 3's vignettes (reuse `train.tortoise_vignettes.VIGNETTES`'s title list for consistency — the prompt and the vignette share a title, but never share text, since a prompt that quoted its own training vignette verbatim would test memorisation, not generation). Each prompt is one sentence that puts the title's image in motion. Three worked examples (write the remaining ~29 following this exact shape — an opening clause or fragment, never a complete self-contained sentence, so completion is required):

```json
{
  "note": "Frozen evaluation set C. A THIRD frozen set, built from real Tortoise (the band) song/album titles -- see docs/superpowers/specs/2026-09-28-storytelling-register-stage-design.md section 5. Every prompt is a scene-opening that puts the title's image in motion, not the bare title: a bare title gives a generative model nothing to continue. NEVER pooled with set A (docs/evaluation_prompts.json) or set B (docs/evaluation_prompts_b.json) -- reported beside them, following exactly the rule those two already follow with each other. Do not edit prompts between runs -- that breaks comparability. Add new ones with new ids instead.",
  "design": [
    "Ids are all prefixed 'c-' so no id can collide with set A's or set B's.",
    "Every prompt's title also has a vignette in train/tortoise_vignettes.py, but the PROMPT text and the VIGNETTE text are never the same string -- a prompt that quoted its own training data verbatim would test memorisation, not generation.",
    "The probe tag is 'evocative-continuation' for every prompt in this set: unlike sets A and B, which stress several distinct behaviours, set C exists for one question -- does the model produce evocative, surreal, well-formed continuations from an abstract image, or does it default to a TinyStories attractor.",
    "This set is deliberately smaller than set B and closer to set A's scale: it exists for a qualitative read (Gate 6 of the design spec) as much as a quantitative one, and 45 prompts of manufactured Tortoise-style titles would dilute the real, verified set this project actually has."
  ],
  "prompts": [
    {"id": "c-namesake-01", "probe": "evocative-continuation",
     "text": "The notebook's final entry, undated, said only that the wire"},
    {"id": "c-djed-01", "probe": "evocative-continuation",
     "text": "No two catalogues agreed on the pillar's height, and the fourth one"},
    {"id": "c-coffin-01", "probe": "evocative-continuation",
     "text": "The instructions arrived folded into quarters, in a hand that"}
  ]
}
```

(Complete the `"prompts"` array to ~32 entries, one per title in Task 3's `EXPECTED_TITLES`, `id` following the pattern `c-<short-slug>-01` with a distinct slug per title, every `text` a genuine sentence fragment ending mid-clause.)

- [ ] **Step 2: Write the failing tests, mirroring set B's exactly**

```python
# tests/test_evaluation_prompts_c.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Prompt set C is FROZEN, exactly as sets A and B are. See test_evaluation_prompts_b.py
for the pattern this mirrors."""
import hashlib
import importlib.util
import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
PROMPTS_C = DOCS / "evaluation_prompts_c.json"
PROMPTS_A = DOCS / "evaluation_prompts.json"
PROMPTS_B = DOCS / "evaluation_prompts_b.json"

ID_PREFIX = "c-"
REQUIRED_PROBES = {"evocative-continuation"}

#: Set once the prompt file's final content is written in Step 1 -- compute with
#: `python -c "import json,hashlib; ..."` reading the actual file (or run the digest
#: test once with a dummy value, read the AssertionError's actual value, and paste it
#: here) rather than guessing. Same requirement Task 3's vignette count has: a real
#: measured value, not an invented one.
FROZEN_DIGEST = "<COMPUTE_FROM_THE_REAL_FILE_IN_STEP_1>"
FROZEN_COUNT = 32


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
    assert isinstance(data["design"], list) and len(data["design"]) >= 3
    for entry in data["design"]:
        assert entry.strip()


def test_every_prompt_has_the_same_schema_as_sets_a_and_b():
    keys_c = {frozenset(p) for p in _prompts_c()}
    assert keys_c == {frozenset({"id", "probe", "text"})}


def test_ids_are_unique():
    ids = [p["id"] for p in _prompts_c()]
    assert len(ids) == len(set(ids))


def test_no_set_c_id_can_collide_with_set_a_or_set_b():
    ids_c = [p["id"] for p in _prompts_c()]
    ids_a = [p["id"] for p in _prompts_a()]
    ids_b = [p["id"] for p in _prompts_b()]
    assert all(pid.startswith(ID_PREFIX) for pid in ids_c)
    assert not any(pid.startswith(ID_PREFIX) for pid in ids_a)
    assert not any(pid.startswith(ID_PREFIX) for pid in ids_b)
    assert not (set(ids_a) | set(ids_b)) & set(ids_c)


def test_no_prompt_text_is_shared_with_set_a_or_set_b():
    texts_c = {p["text"] for p in _prompts_c()}
    assert not texts_c & {p["text"] for p in _prompts_a()}
    assert not texts_c & {p["text"] for p in _prompts_b()}


def test_no_prompt_text_matches_its_own_tortoise_vignette_verbatim():
    """A prompt that reproduced its training vignette would test memorisation, not
    generation -- the whole point this set exists for."""
    from train.tortoise_vignettes import VIGNETTES
    vignette_texts = {text for _title, text in VIGNETTES}
    for p in _prompts_c():
        assert p["text"] not in vignette_texts


def test_no_prompt_is_empty_or_whitespace():
    for p in _prompts_c():
        assert p["text"].strip()


def test_every_prompt_is_a_fragment_not_a_bare_title():
    """A bare title like 'Djed' gives a generative model nothing to continue -- every
    entry must be a real sentence fragment with enough words to establish a scene."""
    for p in _prompts_c():
        assert len(p["text"].split()) >= 8, (
            f"{p['id']}: only {len(p['text'].split())} words, looks like a bare title "
            f"rather than a scene-opening"
        )


def test_no_prompt_text_is_duplicated_within_the_set():
    texts = [p["text"] for p in _prompts_c()]
    assert len(texts) == len(set(texts))


def test_every_required_probe_is_present():
    probes = {p["probe"] for p in _prompts_c()}
    assert probes == REQUIRED_PROBES


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
    assert _digest(prompts + [{"id": "c-extra-01", "text": "one more, unplanned"}]) != FROZEN_DIGEST


def test_the_digest_detects_text_moving_between_two_prompts():
    tampered = [dict(p) for p in _prompts_c()]
    tampered[0]["text"], tampered[1]["text"] = tampered[1]["text"], tampered[0]["text"]
    assert _digest(tampered) != FROZEN_DIGEST


def test_the_digest_does_not_depend_on_file_order():
    assert _digest(list(reversed(_prompts_c()))) == FROZEN_DIGEST


def test_the_three_sets_have_pairwise_distinct_digests():
    assert _digest(_prompts_c()) != _digest(_prompts_a())
    assert _digest(_prompts_c()) != _digest(_prompts_b())


def test_sets_a_and_b_are_untouched_by_the_arrival_of_set_c():
    spec_a = importlib.util.spec_from_file_location(
        "_set_a_pins", Path(__file__).resolve().parent / "test_evaluation_prompts.py")
    set_a = importlib.util.module_from_spec(spec_a)
    spec_a.loader.exec_module(set_a)
    assert len(_prompts_a()) == set_a.FROZEN_COUNT
    assert _digest(_prompts_a()) == set_a.FROZEN_DIGEST

    spec_b = importlib.util.spec_from_file_location(
        "_set_b_pins", Path(__file__).resolve().parent / "test_evaluation_prompts_b.py")
    set_b = importlib.util.module_from_spec(spec_b)
    spec_b.loader.exec_module(set_b)
    assert len(_prompts_b()) == set_b.FROZEN_COUNT
    assert _digest(_prompts_b()) == set_b.FROZEN_DIGEST
```

- [ ] **Step 3: Run to get the real digest, then fill it in**

Run: `python -m pytest tests/test_evaluation_prompts_c.py -k frozen_not_just -q`
Expected: FAILs with an assertion showing the actual computed digest — copy that exact hex string into `FROZEN_DIGEST` in the test file (this is the same bootstrap-the-pin pattern sets A and B themselves were pinned with; not a placeholder, a value read directly off a real run against the file written in Step 1).

- [ ] **Step 4: Run the full test file**

Run: `python -m pytest tests/test_evaluation_prompts_c.py -q`
Expected: PASS (17 tests)

- [ ] **Step 5: Register set C in `score_behaviour.py`**

Find `PROMPT_SETS` (the dict with keys `"a"` and `"b"`) and add:

```python
    "c": PromptSet(
        key="c", path=ROOT / "docs" / "evaluation_prompts_c.json", suffix="-setC",
        description="set C (docs/evaluation_prompts_c.json, ~32 prompts) -- built from "
                    "real Tortoise (the band) song/album titles, one probe "
                    "('evocative-continuation') for whether the model produces "
                    "surreal/imagistic continuations or defaults to a TinyStories "
                    "attractor; reported beside sets A and B and never pooled with "
                    "either"),
```

- [ ] **Step 6: Verify `score_behaviour.py` can load it**

Run: `python -c "
from scripts.score_behaviour import get_prompt_set, load_prompts
ps = get_prompt_set('c')
prompts = load_prompts(ps.path)
print(len(prompts), 'prompts loaded for set C')
"`
Expected: prints `32 prompts loaded for set C` (adjust for whatever `FROZEN_COUNT` actually is once Step 1's file is final).

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -q`
Expected: full suite green, plus the new 17 tests.

- [ ] **Step 8: Commit**

```bash
git add docs/evaluation_prompts_c.json tests/test_evaluation_prompts_c.py scripts/score_behaviour.py
git commit -m "feat(eval): a third frozen prompt set, built from real Tortoise titles

Set C follows set B's exact precedent: its own c- id prefix, its own
frozen digest, never pooled with sets A or B. One probe
(evocative-continuation) -- the instrument for whether the model
actually earns the project's namesake, since the corpus slice alone
(Task 3) is far too small to move any aggregate metric."
```

---

## Task 5: The separator-density gate

**Files:**
- Create: `scripts/check_separator_density.py`
- Test: `tests/test_check_separator_density.py`

**Interfaces:**
- Consumes: nothing new (reads a blend file and a manifest already produced by the existing pipeline).
- Produces: `separator_density(blend_path: Path, manifest_path: Path) -> float` (tokens per document separator); `main()` exits non-zero if the density exceeds a declared ceiling.

**Deliberate deviation from the design spec's exact wording, recorded here rather than silently:** the spec's Gate 1 describes comparing against "the currently-shipped blend's own density" as a relative ±25% band. That comparison needs the currently-shipped `blend.txt` itself, which is gitignored and not reconstructable from the committed `docs/measurements/blend_manifest.json` (which records per-source token/word counts, not document counts). Instead, this gate checks the NEW blend against an absolute ceiling derived from the design decision that produced it: PG19 chunks target 2,000-4,000 words (~2,600-5,200 tokens at this project's measured ~1.3-1.5 tokens/word range) per document, and every other registered source's documents are shorter than that. A ceiling of **6,000 tokens per separator**, comfortably above the top of that range, catches a chunking regression (e.g. the chunker silently returning whole books) while not tripping on the deliberately long PG19 chunks themselves.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_check_separator_density.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
import json
from pathlib import Path

import pytest

from scripts.check_separator_density import separator_density, DENSITY_CEILING_TOKENS_PER_SEP


def _write_blend(tmp_path, n_docs, words_per_doc):
    lines = []
    for _ in range(n_docs):
        lines.append(" ".join(["word"] * words_per_doc))
        lines.append("</s>")
        lines.append("")
    path = tmp_path / "blend.txt"
    path.write_text("\n".join(lines))
    return path


def _write_manifest(tmp_path, total_tokens):
    path = tmp_path / "blend_manifest.json"
    path.write_text(json.dumps({"total_emitted_tokens": total_tokens}))
    return path


def test_separator_density_computes_tokens_per_separator(tmp_path):
    blend = _write_blend(tmp_path, n_docs=10, words_per_doc=100)
    manifest = _write_manifest(tmp_path, total_tokens=1300)  # 10 docs, 130 tok/doc
    assert separator_density(blend, manifest) == pytest.approx(130.0)


def test_separator_density_raises_on_zero_separators(tmp_path):
    blend = tmp_path / "blend.txt"
    blend.write_text("no separators anywhere in this file\n" * 50)
    manifest = _write_manifest(tmp_path, total_tokens=500)
    with pytest.raises(ValueError, match="no document separators"):
        separator_density(blend, manifest)


def test_main_exits_zero_when_density_is_under_the_ceiling(tmp_path, capsys):
    from scripts.check_separator_density import main
    blend = _write_blend(tmp_path, n_docs=100, words_per_doc=500)  # ~500 tok/doc, well under
    manifest = _write_manifest(tmp_path, total_tokens=50_000)
    rc = main(["--blend", str(blend), "--manifest", str(manifest)])
    assert rc == 0


def test_main_exits_nonzero_when_density_exceeds_the_ceiling(tmp_path, capsys):
    from scripts.check_separator_density import main
    blend = _write_blend(tmp_path, n_docs=2, words_per_doc=1)
    manifest = _write_manifest(
        tmp_path, total_tokens=2 * (DENSITY_CEILING_TOKENS_PER_SEP + 1000))
    rc = main(["--blend", str(blend), "--manifest", str(manifest)])
    assert rc == 1
    out = capsys.readouterr().out + capsys.readouterr().err
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_check_separator_density.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.check_separator_density'`

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Gate 1 of the storytelling-register stage: catch a document-boundary regression.

PG19's rows are whole novels; scripts/fetch_corpus.py chunks each into ~2000-4000-word
paragraph-aligned sub-documents (see Task 2) so the end-of-document separator stays at a
sane frequency instead of firing once per book. This script measures the REAL frequency on
an assembled blend and refuses to proceed if it is far outside what that chunking design
implies -- catching, for instance, a chunker silently returning whole unsplit books.

Deliberately an ABSOLUTE ceiling, not a relative comparison against the currently-shipped
blend: that blend's raw text is gitignored and not reconstructable from the committed
manifest (which records per-source token/word counts, not document counts). See this
script's entry in docs/superpowers/plans/2026-09-28-storytelling-register-stage.md Task 5
for the full reasoning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_corpus import DOCUMENT_SEPARATOR  # noqa: E402

#: Ceiling on tokens per document separator. PG19 chunks target 2000-4000 words
#: (~2600-5200 tokens at this project's measured ~1.3-1.5 tokens/word range across its
#: nine/ten existing sources); 6000 sits comfortably above that range so the gate does
#: not trip on the deliberately long PG19 chunks themselves, while still catching a
#: chunker regression that lets whole unsplit novels (tens of thousands of words) through.
DENSITY_CEILING_TOKENS_PER_SEP = 6000


def separator_density(blend_path: Path, manifest_path: Path) -> float:
    """Tokens per document separator in ``blend_path``, using ``manifest_path``'s total.

    Counts literal ``</s>`` lines directly (cheap; no tokenizer needed) and divides the
    manifest's already-measured total token count by that count -- reusing the exact
    total scripts/blend_corpus.py itself computed, rather than re-tokenizing.
    """
    separators = 0
    with Path(blend_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip() == DOCUMENT_SEPARATOR:
                separators += 1
    if separators == 0:
        raise ValueError(
            f"{blend_path} has no document separators at all -- either it is empty or "
            f"something upstream stopped writing them"
        )
    manifest = json.loads(Path(manifest_path).read_text())
    total_tokens = manifest["total_emitted_tokens"]
    return total_tokens / separators


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blend", type=Path, default=ROOT / "artifacts" / "corpus" / "blend.txt")
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "docs" / "measurements" / "blend_manifest.json")
    p.add_argument("--ceiling", type=int, default=DENSITY_CEILING_TOKENS_PER_SEP)
    args = p.parse_args(argv)

    density = separator_density(args.blend, args.manifest)
    print(f"separator density: {density:,.1f} tokens/separator (ceiling: {args.ceiling:,})")
    if density > args.ceiling:
        print(
            f"\nGATE 1 FAILED: {density:,.1f} tokens/separator exceeds the {args.ceiling:,} "
            f"ceiling. This almost always means a source's documents are far longer than "
            f"intended -- check whether PG19's CHUNKED_RENDERERS actually ran (a source "
            f"falling through to the un-chunked path would put whole novels between "
            f"separators). Do not raise the ceiling to force a pass: it exists to catch "
            f"exactly this.", file=sys.stderr)
        return 1
    print("Gate 1 passed: separator density is within the declared ceiling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify all tests pass**

Run: `python -m pytest tests/test_check_separator_density.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: full suite green.

- [ ] **Step 6: Commit**

```bash
git add scripts/check_separator_density.py tests/test_check_separator_density.py
git commit -m "feat(corpus): Gate 1 -- catch a document-separator density regression

Checks the assembled blend's tokens-per-separator against an absolute
ceiling derived from PG19's chunk-size design (Task 2), since the
spec's originally-proposed relative comparison against the currently
shipped blend needs raw text that is gitignored and unreconstructable."
```

---

## Task 6: Re-settle every share at the new budget, with a raised cap for the flavour-class sources

**Files:**
- Modify: `train/corpus.py` (`target_share` and `upsample` values across all 15 registered sources)
- Modify: `docs/measurements/corpus_availability.json` (regenerated)

**Interfaces:** none new — this task's deliverable is evidence and a decision, matching `docs/superpowers/plans/2026-08-13-corpus-assembly.md`'s Task 3.

**The mechanism, stated precisely (from this plan's own brainstorming — see the spec's history and this project's `docs/measurements/external-*.md` for why it matters).** At a ~2.5B budget, a scarce source's ACHIEVABLE tokens (`available x upsample`) stay roughly fixed regardless of the budget, because `available` doesn't grow — only `upsample` and the budget can move. Keeping today's registry SHARES (percentages) at 6.25x the budget would demand ~6.25x the upsample for every source, which the standing 4x cap cannot supply for the scarce curated slices. The user's explicit decision (recorded in this plan's spec-writing conversation): raise the working cap for exactly seven deliberately-small, flavour-class sources — **spine, folklore, weird, flavour, procedural, poetry, tortoise** — to **12x** for this run, rather than accept their percentage share collapsing toward zero (the mechanism that made Stage A, with zero curated content, measurably regress narrative coherence). Every other source (tinystories, gutenberg_children, wikipedia_simple, dialogue, mission, longform, pg19) keeps the standard 4x cap.

- [ ] **Step 1: Fetch pg19 and tortoise**

Run: `python scripts/check_disk_space.py` — confirm exit 0 (adjust `DEFAULT_REQUIRED_GB` upward first if it does not already account for PG19's scale; PG19's train split streams, so disk cost is the PREPARED text this pipeline writes, not the full remote dataset).

Run:
```bash
python scripts/fetch_corpus.py --source pg19 --limit-rows 30000
python scripts/fetch_corpus.py --source tortoise
```
(`--limit-rows` bounds the fetch cost; PG19's ~28,600-book train split streamed in full would massively overshoot what a ~2.5B-token blend could ever use even at a large PG19 share — 30,000 is a starting bound to re-check once Step 2's chunked word count is known, and to raise only if Step 3 shows PG19 short of its needed share.)

- [ ] **Step 2: Prepare and measure**

```bash
python scripts/prepare_corpus.py --source pg19
python scripts/prepare_corpus.py --source tortoise
python scripts/measure_corpus.py --budget 2500000000 --upsample-cap 12
```

Record the full table. Expect PG19 and tortoise to show as `available=0` in the FIRST run if they aren't yet assigned a nonzero `target_share` — `measure_corpus.py`'s gate only reports a shortfall for a source whose `required_tokens` (from a nonzero share) exceeds what's achievable, so a `target_share=0.0` source (as both are registered in Tasks 2/3) will show 0 required and pass trivially without informing the settle. Temporarily set both to a plausible nonzero starting share (e.g. `pg19: 0.45`, `tortoise: 0.002`) before this measurement, purely to see their real availability numbers in the table — the actual settled values come from Step 3.

- [ ] **Step 3: Settle the shares so the gate exits 0**

Adjust `target_share` values in `train/corpus.py` and `upsample` values for the seven named flavour-class sources, obeying these rules:

- Shares must still sum to exactly 1.0.
- **spine, folklore, weird, flavour, procedural, poetry, tortoise** may use up to **12x** upsample (not the standard 8x hard cap either — this run's OWN declared working limit for this named group only).
- **tinystories, gutenberg_children, wikipedia_simple, dialogue, mission, longform, pg19** stay at the existing **4x** working limit.
- PG19 and `longform` (FineWeb-Edu) together should form a large minority-to-majority share per the design spec's "majority narrative" decision — PG19 as the primary vocabulary/register driver, `longform` sized so Stage A's proven loss/commonsense benefit isn't lost, but neither source needs to hit an exact pre-declared number: settle both from real measured availability, same as everything else.
- Do not reduce the *combined* flavour-class share (spine + folklore + weird + flavour + procedural + poetry + tortoise) below whatever the raised 12x cap can actually sustain — if the arithmetic still cannot keep it near its historical ~20-26% combined range even at 12x, stop and report the real ceiling rather than forcing a fit by raising the cap yet again.
- TinyStories stays at or below its current 10% (do not raise it — the whole point of this run is to keep pushing the intervention that already worked).

Record in your report: the shares before (today's registry), the shares after, and the measured number that forced each change — matching corpus-assembly Task 3's own reporting convention.

- [ ] **Step 4: Confirm the gate passes**

Run: `python scripts/measure_corpus.py --budget 2500000000 --upsample-cap 12`
Expected: exit 0, printing "All slices can reach their target share within the cap." (Note: `--upsample-cap 12` here is the REPORTING ceiling `measure_corpus.py` checks against, which must match the raised working limit used for the seven named sources; the remaining eight sources' own `upsample` fields stay at ≤4 regardless of what the CLI flag allows, and Step 5's test enforces that scoping explicitly.)

- [ ] **Step 5: Write the failing test that scopes the raised cap to exactly the named sources**

```python
# add to tests/test_corpus.py
RAISED_CAP_SOURCES = {"spine", "folklore", "weird", "flavour", "procedural", "poetry",
                      "tortoise"}
RAISED_CAP = 12
STANDARD_CAP = 4


def test_only_the_named_flavour_class_sources_use_the_raised_upsample_cap():
    """The raised cap for this run is scoped to exactly seven sources (see Task 6's
    rationale in the storytelling-register-stage plan) -- a source outside that list
    silently exceeding the standard 4x cap would be exactly the unscoped-global-default
    failure mode the plan's own Review Focus names."""
    for name, src in SOURCES.items():
        if name in RAISED_CAP_SOURCES:
            assert src.upsample <= RAISED_CAP, (
                f"{name}: upsample={src.upsample} exceeds this run's raised "
                f"{RAISED_CAP}x cap for flavour-class sources"
            )
        elif src.target_share > 0.0:
            assert src.upsample <= STANDARD_CAP, (
                f"{name}: upsample={src.upsample} exceeds the standard {STANDARD_CAP}x "
                f"working limit, but is not in the list of sources this run "
                f"deliberately raised the cap for"
            )
```

- [ ] **Step 6: Run to verify it fails, then passes after Step 3's edits**

Run: `python -m pytest tests/test_corpus.py -k raised_upsample_cap -q`
Expected: PASS once Step 3's edits are in place (this test is written after the settle, verifying it rather than driving it — the settle itself is a data/evidence task, not a code-first TDD one, matching corpus-assembly Task 3's own precedent).

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass, including `test_target_shares_sum_to_one` against the new values.

- [ ] **Step 8: Commit**

```bash
git add train/corpus.py docs/measurements/corpus_availability.json tests/test_corpus.py
git commit -m "measure: settle all 15 corpus shares at a 2.5B budget; gate exits 0

Raised the working upsample cap to 12x for exactly seven deliberately-
small flavour-class sources (spine, folklore, weird, flavour,
procedural, poetry, tortoise) to keep their combined share from
collapsing toward zero at this much larger budget -- the same dilution
mechanism the data-scale-up spec identified, addressed directly rather
than accepted."
```

---

## Task 7: Build the blend, and run Gate 1 against it

**Files:**
- (No new source files — this task runs the pipeline built in prior tasks.)
- Modify: `docs/measurements/blend_manifest.json` (regenerated, tracked copy)

**Interfaces:** none new.

- [ ] **Step 1: Build the blend**

```bash
python scripts/blend_corpus.py --budget 2500000000
```

Expected: exits 0, prints a per-source table, and reports `total ... tokens against a 2,500,000,000 budget` within a few tenths of a percent — matching this project's own established tolerance (every prior blend has landed within ~0.1-0.15% of its declared budget). Confirm `WARNING: real repetition exceeds the declared upsample` does NOT print for any source (it would mean Task 6's settle has a bug: `plan_blend`'s gate is supposed to make this unreachable).

- [ ] **Step 2: Run Gate 1**

```bash
python scripts/check_separator_density.py
```

Expected: exit 0, "Gate 1 passed". If it fails, the most likely cause (per this script's own error message) is PG19 falling through to the un-chunked renderer path — re-verify Task 2's `CHUNKED_RENDERERS` registration is keyed on the exact `hf_repo` string used in the `pg19` `CorpusSource` entry.

- [ ] **Step 3: Spot-check a real chunk boundary by hand (the Review Focus item)**

```bash
python -c "
from pathlib import Path
text = Path('artifacts/corpus/pg19.txt').read_text(encoding='utf-8', errors='replace')
docs = text.split('</s>')
print(f'{len(docs)} pg19 documents')
# print two adjacent document boundaries for a human to read
print('--- END OF DOC 5 ---')
print(docs[4][-300:])
print('--- START OF DOC 6 ---')
print(docs[5][:300])
"
```
Read the printed excerpt. Confirm the boundary lands at the end of a paragraph (not mid-sentence) and that the transition, while a real topic change, doesn't look like a chunker bug (e.g. repeated text, garbled encoding). Record what you observed in the task's own commit message or a short note — this is the human check the Review Focus item calls for; the unit tests in Task 2 already prove no paragraph is split, this step confirms the RESULT reads sanely to a person.

- [ ] **Step 4: Commit the tracked manifest**

```bash
git add docs/measurements/blend_manifest.json
git commit -m "build: assemble the storytelling-register blend at a 2.5B token budget

Gate 1 (separator density) passed; a hand-read chunk boundary in the
PG19 slice confirmed sane paragraph-aligned splitting."
```

---

## Task 8: Retrain the tokenizer at 48K, and raise the model registry to match

**Files:**
- Modify: `train/sizes.py` (`SIZES["1024"].vocab_size`, rationale)
- Modify: `train/configs/model/tt-tnt-1024.yaml` (`vocab_size: 48000`)
- Test: `tests/test_sizes.py` (verify no change needed, or fix if it hardcodes 32000 for "1024" specifically)

**Interfaces:**
- Consumes: `artifacts/corpus/blend.txt` (Task 7).
- Produces: `artifacts/tokenizer-storyreg/` (a NEW directory — never `artifacts/tokenizer/`, per the Global Constraints).

- [ ] **Step 1: Retrain the tokenizer to a new directory**

```bash
python scripts/build_tokenizer.py --corpus artifacts/corpus/blend.txt --vocab-size 48000
```

This will fail or misbehave as written: `build_tokenizer.py` always writes to the hardcoded `ARTIFACTS / "tokenizer"` (see its `main()`), which this plan's Global Constraints forbid touching. Before running it, make ONE local, uncommitted, temporary edit for this invocation only (or add a `--tokenizer-out` flag if you prefer a permanent, tested capability — the minimal path is described here): change the line `tok_out = ARTIFACTS / "tokenizer"` to read from a new `--tokenizer-out` CLI argument defaulting to the existing path, so every other invocation is unaffected. If you add the flag, write it as a real, tested change (a one-line `argparse` addition plus a test asserting the default is unchanged), not a throwaway edit — this project's own convention is that a script's behavior is defined by its tests, not by a comment saying "temporarily changed."

```bash
python scripts/build_tokenizer.py --corpus artifacts/corpus/blend.txt \
    --vocab-size 48000 --tokenizer-out artifacts/tokenizer-storyreg
```

Expected: `>> Achieved vocabulary: 48,000 (target: 48,000)`, exit 0. If it under-shoots, that means the blend (2.5B tokens across many sources) still didn't have enough distinct subword pairs to reach 48K — re-run with a larger `--vocab-size` cap is the WRONG fix per this script's own existing error message; the right fix is confirming Task 6/7's blend is actually the full 2.5B-token one and not a truncated intermediate.

- [ ] **Step 2: Update the model registry**

In `train/sizes.py`, change `SIZES["1024"]`'s `vocab_size=32000,` to `vocab_size=48000,` and append to its `rationale` string (do not delete the existing history — this project's convention, visible throughout the current text, is to append a dated note rather than rewrite history):

```python
            "TOKENIZER RETRAINED at 48000 tokens on 2026-09-28 for the storytelling-"
            "register stage (see docs/superpowers/specs/2026-09-28-storytelling-"
            "register-stage-design.md): the new blend (deepmind/pg19 plus the "
            "existing curated sources, reweighted) is large and lexically diverse "
            "enough to want more subword units than 32000. tokenizer path for this "
            "line is artifacts/tokenizer-storyreg, NOT the shared artifacts/tokenizer "
            "default every prior checkpoint's header points at -- see train/run.py's "
            "--tokenizer-dir flag (added in this same stage's Task 1)."
```

In `train/configs/model/tt-tnt-1024.yaml`, change line 88 from `vocab_size: 32000` to `vocab_size: 48000`.

- [ ] **Step 3: Verify the anti-drift test still passes**

Run: `python -m pytest tests/test_sizes.py -q`
Expected: PASS — the existing test (per this project's convention: "tests/test_sizes.py asserts the two never drift apart") reads both the registry and the YAML and should now see them agree at 48000 rather than fail on a mismatch. If it fails, read its exact assertion message before editing further: it either hardcodes 32000 somewhere for this specific size (fix by removing the hardcode, deriving from `SIZES["1024"].vocab_size` instead) or it is working exactly as designed and caught a real edit you haven't made yet.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: full suite green. Confirm nothing under `tests/` still asserts `SIZES["1024"].vocab_size == 32000` anywhere (grep first: `grep -rn "1024.*32000\|32000.*1024" tests/`).

- [ ] **Step 5: Commit**

```bash
git add train/sizes.py train/configs/model/tt-tnt-1024.yaml scripts/build_tokenizer.py \
        tests/test_build_tokenizer.py
git commit -m "feat(tokenizer): retrain at 48K for the storytelling-register stage

New tokenizer at artifacts/tokenizer-storyreg (never the shared
artifacts/tokenizer default every prior checkpoint's header depends
on). train/sizes.py's 1024 entry and its YAML both raised to
vocab_size=48000, verified in agreement by the existing anti-drift
test."
```

---

## Task 9: Tokenize the corpus with a stratified split

**Files:**
- (No new source files.)

**Interfaces:** none new.

- [ ] **Step 1: Tokenize**

```bash
python -m train.tokenization --corpus artifacts/corpus/blend.txt \
    --tokenizer artifacts/tokenizer-storyreg \
    --out artifacts/tokens-storyreg \
    --blend-manifest docs/measurements/blend_manifest.json
```

Expected: prints a `TokenStats` with `vocab_size=48000` and a `source_splits` breakdown covering all 15 registered sources with a nonzero `target_share`. This uses the STRATIFIED split (`--blend-manifest` given), the same choice this project made after discovering the default tail-of-the-whole-stream split silently produces a single-source validation set on a multi-source corpus.

- [ ] **Step 2: Sanity-check the token arrays**

```bash
python -c "
import numpy as np
train = np.load('artifacts/tokens-storyreg/train_ids.npy')
val = np.load('artifacts/tokens-storyreg/val_ids.npy')
print('train', len(train), 'val', len(val), 'max id', int(max(train.max(), val.max())))
assert int(train.max()) < 48000 and int(val.max()) < 48000
print('EOS (id 2) occurrences in train:', int((train == 2).sum()))
assert int((train == 2).sum()) > 0
"
```
Expected: `max id` under 48000; a nonzero EOS count (confirming Task 2/5's separator work actually reached the token stream, the same verification this project ran after the original document-boundary fix).

- [ ] **Step 3: Commit nothing new**

This task writes only to `artifacts/`, which is gitignored — there is no commit for this task beyond noting the command and its output in the next task's report. If you added a `--tokenizer-out` flag to `build_tokenizer.py` in Task 8 and it wasn't yet committed, commit it now instead of leaving it uncommitted.

---

## Task 10: Train two seed replicates

**Files:**
- (No new source files.)

**Interfaces:** none new.

**Hardware safety:** lease before touching a device. `gozer run --chips 4 --who "claude:storytelling-register" --reason "storyreg training, 2 seeds" -- <command>`, or `gozer acquire --chips 4 ...` for a lease spanning both training invocations, released with `gozer release <lease>` after Task 11 also finishes reading these checkpoints (conversion needs no device, but keeping the lease through both seeds avoids re-queueing).

- [ ] **Step 1: Compute the step count for a ~2.5B-token, one-epoch run**

`steps = total_train_tokens // (batch_size * seq_len)`. With `batch_size=64` and `seq_len=512` (the "1024" size's `max_sequence_length`, unchanged per the spec's decision to hold context at 512), and `total_train_tokens` read from Task 9's `TokenStats.train_tokens`: `steps = train_tokens // (64 * 512)`. Compute this from the REAL number Task 9 printed, not from the 2.5B target (the achieved total will differ slightly, per every prior blend's ~0.1-0.15% variance).

- [ ] **Step 2: Seed 1 — the project's own default seed, for continuity with every prior committed baseline**

```bash
gozer run --chips 4 --who "claude:storytelling-register" --reason "storyreg seed 5489" -- \
python train/run.py \
    --size 1024 --ddp 4 --model-impl python \
    --tokenizer-dir artifacts/tokenizer-storyreg \
    --tokens-dir artifacts/tokens-storyreg \
    --config train/configs/nanollama3_bpe_v2.yaml \
    --lr-schedule cosine \
    --seed 5489 \
    --steps <computed_from_step_1> \
    --save-every <steps // 10, rounded> \
    --val-every <steps // 20, rounded> \
    --checkpoint-dir artifacts/checkpoints-storyreg-s5489
```

Confirm before trusting the run: the startup log shows `stochastic_rounding: True` (from `nanollama3_bpe_v2.yaml`), and the printed `tokenizer_dir` line matches `artifacts/tokenizer-storyreg`, not the shared default — this is exactly what Task 1 exists to make possible and verifiable.

- [ ] **Step 3: Seed 2 — a genuinely different seed, per the spec's "at least 2 seeds" gate**

```bash
gozer run --chips 4 --who "claude:storytelling-register" --reason "storyreg seed 20260815" -- \
python train/run.py \
    --size 1024 --ddp 4 --model-impl python \
    --tokenizer-dir artifacts/tokenizer-storyreg \
    --tokens-dir artifacts/tokens-storyreg \
    --config train/configs/nanollama3_bpe_v2.yaml \
    --lr-schedule cosine \
    --seed 20260815 \
    --steps <same as seed 1> \
    --save-every <same> \
    --val-every <same> \
    --checkpoint-dir artifacts/checkpoints-storyreg-s20260815
```

(`20260815` matches a seed already used in this project's TinyStories-reduction experiment, chosen for the same reason: an arbitrary-but-already-precedented second seed, not a freshly invented one.)

- [ ] **Step 4: Verify both checkpoints' headers before trusting anything downstream**

```bash
python -c "
from convert.checkpoint_reader import read_checkpoint_meta
for seed in ('s5489', 's20260815'):
    header, _ = read_checkpoint_meta(f'artifacts/checkpoints-storyreg-{seed}/tt_tnt_step_LATEST.pkl')
    print(seed, 'vocab_size', header['vocab_size'], 'tokenizer_dir', header['tokenizer_dir'])
    assert header['vocab_size'] == 48000
    assert header['tokenizer_dir'] == 'artifacts/tokenizer-storyreg'
"
```
(Replace `tt_tnt_step_LATEST.pkl` with the actual highest-step filename via `train.checkpoint.latest_checkpoint`, per this project's existing helper.) This is the direct proof that Task 1's fix reached a real run: both fields must read the NEW values, not the legacy `32000`/`artifacts/tokenizer` defaults.

- [ ] **Step 5: Release the lease**

```bash
gozer release <lease_id>
```

- [ ] **Step 6: Record, do not commit**

No source files change. Record both runs' final train/val loss, wall clock, and step count in the next task's report — `artifacts/checkpoints-storyreg-*` is gitignored.

---

## Task 11: Convert both checkpoints to HF, and establish this line's own seed-noise floor

**Files:**
- (No new source files.)

**Interfaces:** none new.

- [ ] **Step 1: Convert both seeds**

```bash
python -c "
from convert.to_hf import convert_checkpoint
from train.checkpoint import latest_checkpoint
for seed in ('s5489', 's20260815'):
    ckpt = latest_checkpoint(f'artifacts/checkpoints-storyreg-{seed}')
    convert_checkpoint(ckpt, 'artifacts/tokenizer-storyreg', f'artifacts/hf-storyreg-{seed}')
"
```

- [ ] **Step 2: Spot-check each converted model loads and answers a basic question**

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
for seed in ('s5489', 's20260815'):
    d = f'artifacts/hf-storyreg-{seed}'
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForCausalLM.from_pretrained(d)
    ids = tok('Q: What is the capital of France?\n\nAnswer:', return_tensors='pt').input_ids
    out = model.generate(ids, max_new_tokens=10, do_sample=False)
    print(seed, tok.decode(out[0]))
"
```
Read the output. This model has no dialogue/Q&A fine-tuning at this stage (that was the SFT line's job, not this pretraining stage's), so a coherent factual answer is NOT expected — the check here is that generation runs without error and produces plausible English tokens, not gibberish or an immediate crash. Note whatever it actually says in the task report rather than asserting a pass/fail on content.

- [ ] **Step 3: Establish the seed-noise floor for THIS line**

This project's committed `docs/measurements/seed-noise-floor.json` was derived from the OLD tokenizer/corpus (tt-tnt-v3 vs. v5) and does not describe this new line — the whole reason Task 6's spec called for "at least 2 seeds" is to derive a floor specific to this corpus/tokenizer, not borrow the old one.

```bash
python scripts/evaluate.py --model artifacts/hf-storyreg-s5489 \
    --against artifacts/hf-storyreg-s20260815 \
    --prompt-set a --skip-trajectory \
    --out-dir docs/measurements
python scripts/evaluate.py --model artifacts/hf-storyreg-s5489 \
    --against artifacts/hf-storyreg-s20260815 \
    --prompt-set b --skip-trajectory \
    --out-dir docs/measurements
```

This pairs the two SAME-corpus, SAME-recipe, DIFFERENT-seed runs against each other — by definition a seed-only comparison, which is exactly what a noise floor is. Record the resulting behavioural sds and the loss delta's sd as this line's own floor figures (do not overwrite `docs/measurements/seed-noise-floor.json`, which belongs to the old line — write these under new, clearly-named files, e.g. `docs/measurements/seed-noise-floor-storyreg.json`, following this project's convention of never silently repurposing a filename that another line's measurements already depend on).

- [ ] **Step 4: Commit the new floor measurement docs**

```bash
git add docs/measurements/seed-noise-floor-storyreg.json \
        docs/measurements/behaviour-storyreg-s5489-vs-s20260815-setA.md \
        docs/measurements/behaviour-storyreg-s5489-vs-s20260815-setA.json \
        docs/measurements/behaviour-storyreg-s5489-vs-s20260815-setB.md \
        docs/measurements/behaviour-storyreg-s5489-vs-s20260815-setB.json
git commit -m "measure: seed-noise floor for the storytelling-register line

Two same-corpus, same-recipe, different-seed runs, paired against each
other -- this line's own floor, since the committed seed-noise-floor.json
describes the old 32K-tokenizer/tokens-v4 line and does not transfer."
```

---

## Task 12: Gates 4-6 — the register comparison against the current designated checkpoint

**Files:**
- (No new source files.)
- Modify: `docs/corpus_blend.md` (describes the new blend, same honest-banner pattern the current one uses)

**Interfaces:** none new.

- [ ] **Step 1: Gate 4 — register comparison, both seeds, both prompt sets, against `tt-tnt-1024`**

```bash
for seed in s5489 s20260815; do
  python scripts/evaluate.py --model artifacts/hf-storyreg-$seed \
      --against artifacts/hf-tt-tnt-1024 \
      --prompt-set a --skip-trajectory --out-dir docs/measurements
  python scripts/evaluate.py --model artifacts/hf-storyreg-$seed \
      --against artifacts/hf-tt-tnt-1024 \
      --prompt-set b --skip-trajectory --out-dir docs/measurements
done
```

`--skip-trajectory` because this run's `val_losses.jsonl` windows are not the same window as `tt-tnt-1024`'s recorded trajectory (this line trains at whatever `train_ids`/512 gives; confirm the windows genuinely match before ever dropping `--skip-trajectory` — `scripts/evaluate.py`'s own window guard refuses a mismatch, so trust its refusal rather than overriding it).

Read the story-frame and lexical-habit collapse rates and the TinyStories register margin for BOTH seeds. Per the spec's Gate 4: **both** the floor-ratio gate and the paired-minimum-detectable-difference gate must clear, on **both** seeds, using this line's own floor from Task 11 Step 3 (not the old `seed-noise-floor.json`). A result clearing on only one seed, or clearing the ratio gate but not the MDE gate (or vice versa), is reported as `NOT INTERPRETABLE` / `below paired detection` exactly as `scripts/evaluate.py` already labels such cases — do not round an ambiguous result up into a finding.

- [ ] **Step 2: Gate 5 — guardrails**

From the same reports: termination rate, 4-gram repeat rate, StoryCloze accuracy (run `scripts/eval_storycloze.py` against both `artifacts/hf-storyreg-*` directories and the `tt-tnt-1024` baseline if it isn't already covered by the above). Any guardrail regression beyond ITS floor (from Task 11) is reported as a real cost alongside whatever register gain exists — never silently dropped.

- [ ] **Step 3: Gate 6 — qualitative read on set C**

```bash
python scripts/evaluate.py --try-file <(python -c "
import json
prompts = json.load(open('docs/evaluation_prompts_c.json'))['prompts']
for p in prompts:
    print(p['text'])
") --model artifacts/hf-storyreg-s5489 --out-dir scratch/storyreg-set-c-read
```

(Or use `score_behaviour.py --prompt-set c` directly against both seed checkpoints for a scored, not just ad-hoc, read.) Read every completion, watching for two distinct things, per the spec's §3 risk note: (1) does the model take an abstract image ("the pillar's backbone", "the coffin instructions") and go somewhere coherent and evocative, or does it default to a TinyStories-shaped continuation regardless of the prompt; and (2) separately, does the *voice* skew archaic/19th-century (period diction, "thee"/"thou", Victorian sentence rhythm) rather than merely richer-vocabulary — PG19's own dataset card flags this as a real risk, and "less TinyStories-like" would also move in that direction for the wrong reason. Report both observations distinctly; neither has a numeric pass/fail threshold — state them as qualitative findings, exactly as the spec's Gate 6 calls for.

- [ ] **Step 4: Update `docs/corpus_blend.md`**

Add a new section (or a new banner, following the existing document's own pattern for the long-context-corpus experiment) stating: this is a new, twelve-plus-source blend (list the real count once Task 6 is final), no published model has trained on it, and pointing to this stage's spec and this task's measurement files for what was actually found. Do not claim a promotion decision here — that is explicitly out of scope (see Global Constraints).

- [ ] **Step 5: Write the final measurement summary**

Create `docs/measurements/storytelling-register-stage-summary.md` stating, plainly: the settled shares (Task 6), the achieved token budget (Task 7), the tokenizer's achieved vocabulary (Task 8), both seeds' final losses (Task 10), the Gate 4/5 verdicts with their exact numbers and floor ratios, and the Gate 6 qualitative read. State explicitly whether this is a candidate worth a human promotion decision, or a documented null — per this project's own standing rule, a confirmed null is a valid, reportable outcome, not a failure to paper over.

- [ ] **Step 6: Commit**

```bash
git add docs/corpus_blend.md docs/measurements/storytelling-register-stage-summary.md \
        docs/measurements/behaviour-storyreg-*
git commit -m "measure: storytelling-register stage vs tt-tnt-1024, gates 4-6

Full register comparison (both seeds, both frozen prompt sets) against
the current designated checkpoint, guardrail check, and a qualitative
read on the new Tortoise-title prompt set. Promotion, if any, is a
separate human decision -- docs/current_model.json is unchanged."
```
