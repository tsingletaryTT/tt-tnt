#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Does the tool-calling checkpoint actually emit well-formed tool calls?

CPU-only (transformers, no ttnn/ttml/device). Three questions, in increasing strictness:

1. **Emission rate** -- does a `<tool_call>...</tool_call>` block appear at all?
2. **Parse rate** -- does `train.tool_calling.parse_tool_call` (the same function that
   inverts what training rendered, and the same shape vllm's hermes parser scans for)
   return a tool + arguments from it?
3. **Schema-valid rate** -- does the parsed call name a REGISTERED tool with exactly its
   required arguments and legal enum values (`validate_example`)? A call that parses as
   JSON but invents a tool name or drops an argument would be accepted by a lenient
   consumer and is not a success.

Held to a control the way this project holds everything else: the same battery is run
against the WARM-START base checkpoint, which never saw a tool call. Its emission rate is
the floor -- if the base already emits these, the training taught nothing. This is the same
control shape `improv-stage1` used (a no-think arm emitting blocks 0% of the time is what
made 98% adherence meaningful).

    python scripts/eval_tool_calling.py --model artifacts/hf-... --control artifacts/hf-...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.tool_calling import (  # noqa: E402
    TOOLS,
    ToolCallExample,
    parse_tool_call,
    validate_example,
)

#: Held-out-ish probe questions. Deliberately a MIX: some appear in the hand-authored seeds
#: (so this measures reproduction) and some do not (so it measures generalisation to an
#: unseen question). Which is which is recorded per question rather than averaged away --
#: a model that only emits tool calls for questions it memorised has not learned the format.
PROBES = [
    ("What is the capital of France?", "seen"),
    ("How many legs does a spider have?", "seen"),
    ("Why is the sky blue?", "seen"),
    ("What is the tallest mountain?", "seen"),
    ("What is the capital of Portugal?", "unseen"),
    ("Why do rivers bend?", "unseen"),
    ("How many strings does a violin have?", "unseen"),
    ("What is the smallest bone in the body?", "unseen"),
]


def classify_completion(text: str, question: str) -> Dict[str, object]:
    """Score ONE completion against the three gates. Extracted from the driver deliberately.

    This project has now been bitten twelve times by decision logic that lived only inside a
    composition function no test imported -- CLAUDE.md's reach-dial entry records three
    one-line mutations inside such a function each rewriting a published claim while 1,505
    tests passed. The counting rules here (what counts as emitted vs parsed vs schema-valid)
    are exactly that kind of logic, so they live in a function a test can call directly with
    hand-written strings and no model.

    Returns a dict with `emitted`, `parsed`, `schema_valid` (bools) and `tool` (str or None).
    The three are deliberately NOT independent: schema_valid implies parsed implies emitted
    is *usually* true, but a completion could in principle parse without the literal tag if
    parse_tool_call's regex ever loosened -- so each is measured on its own rather than
    inferred from the one below it.
    """
    emitted = "<tool_call>" in text
    call = parse_tool_call(text)
    if call is None:
        return {"emitted": emitted, "parsed": False, "schema_valid": False, "tool": None}
    name, args = call
    schema_valid = False
    try:
        validate_example(ToolCallExample(
            question=question, tool=name,
            arguments={k: str(v) for k, v in (args or {}).items()},
        ))
        schema_valid = True
    except (ValueError, TypeError, AttributeError):
        pass
    return {"emitted": emitted, "parsed": True, "schema_valid": schema_valid, "tool": name}


def evaluate(model_dir: Path, n_samples: int, max_new_tokens: int, seed: int) -> Dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32)
    model.eval()
    torch.manual_seed(seed)

    emitted = parsed_ok = schema_ok = 0
    total = 0
    tool_counts: Counter = Counter()
    per_question: List[Dict] = []
    samples: List[Dict] = []

    for question, provenance in PROBES:
        prompt = f"Q: {question}\nAnswer:"
        ids = tok(prompt, return_tensors="pt").input_ids
        q_emitted = q_parsed = q_schema = 0
        for i in range(n_samples):
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id or 0,
                )
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            total += 1
            verdict = classify_completion(text, question)
            if verdict["emitted"]:
                emitted += 1
                q_emitted += 1
            if verdict["parsed"]:
                parsed_ok += 1
                q_parsed += 1
                tool_counts[str(verdict["tool"])] += 1
            if verdict["schema_valid"]:
                schema_ok += 1
                q_schema += 1
            if i == 0:
                samples.append({"question": question, "provenance": provenance,
                                "completion": text[:400]})
        per_question.append({
            "question": question, "provenance": provenance, "n": n_samples,
            "emitted": q_emitted, "parsed": q_parsed, "schema_valid": q_schema,
        })

    return {
        "model": str(model_dir),
        "n_total_generations": total,
        "emission_rate": emitted / total if total else 0.0,
        "parse_rate": parsed_ok / total if total else 0.0,
        "schema_valid_rate": schema_ok / total if total else 0.0,
        "tool_distribution": dict(tool_counts),
        "distinct_tools_used": len(tool_counts),
        "registered_tools": sorted(TOOLS),
        "per_question": per_question,
        "first_sample_per_question": samples,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--control", type=Path, default=None,
                    help="warm-start base, which never saw a tool call -- its emission "
                         "rate is the floor that makes the model's number mean anything")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    result = {"model": evaluate(args.model, args.n_samples, args.max_new_tokens, args.seed)}
    if args.control is not None:
        result["control"] = evaluate(args.control, args.n_samples, args.max_new_tokens,
                                     args.seed)

    for key in ("model", "control"):
        if key not in result:
            continue
        r = result[key]
        print(f"\n=== {key.upper()}: {r['model']} ===")
        print(f"  emission     {r['emission_rate']:.1%}")
        print(f"  parse        {r['parse_rate']:.1%}")
        print(f"  schema-valid {r['schema_valid_rate']:.1%}")
        print(f"  tools used   {r['distinct_tools_used']}/{len(r['registered_tools'])} "
              f"{r['tool_distribution']}")
        for pq in r["per_question"]:
            print(f"    [{pq['provenance']:6s}] {pq['question'][:44]:46s} "
                  f"emit {pq['emitted']}/{pq['n']}  schema {pq['schema_valid']}/{pq['n']}")

    if "control" in result:
        m, c = result["model"], result["control"]
        print(f"\nschema-valid rate: model {m['schema_valid_rate']:.1%} vs "
              f"control {c['schema_valid_rate']:.1%}  "
              f"(delta {m['schema_valid_rate'] - c['schema_valid_rate']:+.1%})")

    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
