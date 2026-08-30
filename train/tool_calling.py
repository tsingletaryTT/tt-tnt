# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tool-calling response modes: teaching the model to name HOW it is answering.

WHY THIS EXISTS
---------------
The reach dial taught the model to name a property of what it was about to say (near/mid/far)
before saying it. The skits schema taught it to name a *role* (offer/accept/add/stakes/
handback) each turn plays in a scene. This is the same idea applied to register: instead of
just answering a question, the model calls a real tool -- through the SAME hermes tool-call
protocol already wired into serving (`--tool-call-parser hermes`, `docs/serving-with-tt-kernel.md`)
-- naming which mode it is answering in, with structured arguments that carry more than a bare
tag would.

Four tools, deliberately small and orthogonal:

    factual_response(answer, confidence: high|low)
    witty_response(answer, technique: pun|wordplay|reference)
    absurdist_response(answer, logic)          -- logic states the fake internal reasoning
    misunderstood_question(interpreted_as, answer)  -- states what it thought was asked

THE TWO-LAYER CORPUS
--------------------
This project's standing discipline is "derive from what's actually in the corpus" (skit turns,
the reach dial, the dialogue slice itself). Nothing in this corpus is naturally "a witty wrong
answer to a trivia question," so this is the first slice that is substantially *authored* rather
than *derived* -- acknowledged rather than hidden. Two layers, blended:

1. HAND_AUTHORED_SEEDS below: ~100 examples, written in a deliberately eclectic voice (Brautigan's
   gentle whimsy, Douglas Adams' deadpan absurdist logic, Zork/AD&D flavor-text deadpan, Gertrude
   Stein's repetition, Ginsberg's incantatory cadence, Borges' labyrinthine self-reference, Tom
   Robbins' maximalist metaphor) -- inspired by that lineage's TONE and LOGIC, never paraphrasing
   or reproducing any specific passage from any of them, several of whom are living authors and
   active estates.
2. `mine_factual_pairs` + `derive_templated_variants`: real short Question/Answer pairs mined
   directly from `artifacts/corpus/dialogue.txt` (the databricks-dolly-15k slice already in the
   corpus), expanded into witty/absurdist/misunderstood variants by simple, declared rule-based
   templates -- mechanical humor, not claimed to be as good as the hand-authored core, but
   genuinely derived from real corpus content rather than invented from nothing.

RENDERED FORMAT
---------------
Training text is `Q: {question}\nAnswer:` as the prompt (matching the existing dialogue-slice
convention this project already established) followed by a completion that is a real hermes-style
tool call:

    <tool_call>
    {"name": "witty_response", "arguments": {"answer": "...", "technique": "pun"}}
    </tool_call>

This is not a training-only convention invented for this file: it is the EXACT text shape
`vllm.tool_parsers.hermes_tool_parser` scans for in decoded output (plain substring matching on
the literal `<tool_call>` tag, no special-token dependency -- see the reasoning recorded in
CLAUDE.md for why `hermes` was chosen over `llama3_json`). Training on this literal text is what
makes the served model's tool call real, not simulated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

#: Tool schema: name -> (required argument names, in order) and any enum-constrained arguments.
#: A hand-authored or derived example that doesn't match this exactly is a bug, not a style
#: choice -- caught by validate_example, not left to be caught downstream by a confused parser.
TOOLS: Dict[str, Dict[str, object]] = {
    "factual_response": {
        "required_args": ("answer", "confidence"),
        "enum_args": {"confidence": ("high", "low")},
    },
    "witty_response": {
        "required_args": ("answer", "technique"),
        "enum_args": {"technique": ("pun", "wordplay", "reference")},
    },
    "absurdist_response": {
        "required_args": ("answer", "logic"),
        "enum_args": {},
    },
    "misunderstood_question": {
        "required_args": ("interpreted_as", "answer"),
        "enum_args": {},
    },
}


@dataclass(frozen=True)
class ToolCallExample:
    """One (question, tool call) training pair."""

    question: str
    tool: str
    arguments: Dict[str, str]
    #: "hand" for HAND_AUTHORED_SEEDS, "derived" for mine_factual_pairs/derive_templated_variants
    #: output -- kept on the example itself so a consumer can weight or filter by provenance
    #: without re-deriving it, the same way corpus documents carry their source name.
    provenance: str = "hand"


def validate_example(example: ToolCallExample) -> None:
    """Raise ValueError naming exactly what's wrong. Never silently accept a malformed example --
    a bad tool call trains the model to emit exactly the malformed thing it was given."""
    if example.tool not in TOOLS:
        raise ValueError(
            f"unknown tool {example.tool!r} for question {example.question!r}; "
            f"registered tools: {sorted(TOOLS)}"
        )
    spec = TOOLS[example.tool]
    required = spec["required_args"]
    missing = [a for a in required if a not in example.arguments]
    if missing:
        raise ValueError(
            f"{example.tool}({example.question!r}) missing required argument(s) {missing}; "
            f"needs {required}"
        )
    extra = set(example.arguments) - set(required)
    if extra:
        raise ValueError(
            f"{example.tool}({example.question!r}) has unexpected argument(s) {sorted(extra)}; "
            f"only {required} are declared for this tool"
        )
    for arg_name, allowed in spec["enum_args"].items():
        value = example.arguments[arg_name]
        if value not in allowed:
            raise ValueError(
                f"{example.tool}({example.question!r}).{arg_name}={value!r} not in {allowed}"
            )
    if not example.question.strip():
        raise ValueError("question is empty")
    for arg_name, value in example.arguments.items():
        if not str(value).strip():
            raise ValueError(f"{example.tool}({example.question!r}).{arg_name} is empty")


def render_tool_call(tool: str, arguments: Dict[str, str]) -> str:
    """The exact hermes-parseable text: `<tool_call>\\n{json}\\n</tool_call>`.

    Argument order in the emitted JSON follows TOOLS[tool]["required_args"], not dict insertion
    order -- so two examples for the same tool always serialize their arguments in the same
    order, which matters for a small model learning a fixed emission pattern.
    """
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}; registered tools: {sorted(TOOLS)}")
    ordered = {k: arguments[k] for k in TOOLS[tool]["required_args"]}
    payload = json.dumps({"name": tool, "arguments": ordered}, ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """Inverse of render_tool_call, for scoring served output. Returns (tool, arguments) or None
    if no well-formed `<tool_call>...</tool_call>` block is found. Deliberately permissive about
    surrounding whitespace/text (a served completion may have content before the tag) but strict
    about the JSON inside it -- a malformed JSON payload is a real miss, not something to paper
    over with a lenient parse."""
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "name" not in payload or "arguments" not in payload:
        return None
    return payload["name"], payload["arguments"]


def build_training_text(example: ToolCallExample) -> str:
    """The full text the model is trained to reproduce: prompt + the tool-call completion."""
    return f"Q: {example.question}\nAnswer:{render_tool_call(example.tool, example.arguments)}"


# ---------------------------------------------------------------------------------------
# Layer 1: hand-authored seeds
# ---------------------------------------------------------------------------------------

#: Each entry: (question, tool, arguments). Built as raw tuples rather than ToolCallExample
#: directly so the list stays terse to read and edit; `hand_authored_examples()` below is what
#: actually validates and wraps them.
_HAND_AUTHORED: List[Tuple[str, str, Dict[str, str]]] = [
    # --- capital of France ---
    ("What is the capital of France?", "witty_response",
     {"answer": "F. The capital letter of France is F.", "technique": "wordplay"}),
    ("What is the capital of France?", "witty_response",
     {"answer": "Paris, though on a clear day you can also see it from Texas.", "technique": "reference"}),
    ("What is the capital of France?", "absurdist_response",
     {"answer": "Paris is where the light goes when it's tired of being light. It lies down in "
                "the Seine like a fish that has given up on being caught.",
      "logic": "capitals are where tired things go to rest"}),
    ("What is the capital of France?", "absurdist_response",
     {"answer": "You are in a maze of twisty little boulevards, all alike. The capital of France "
                "is here, but it is dark, and you are likely to be eaten by a Métro.",
      "logic": "when in doubt, you are in a maze"}),
    ("What is the capital of France?", "absurdist_response",
     {"answer": "PARIS. Alignment: Chaotic Beautiful. Population (HP): 2,150,000. Special "
                "ability: once per era, may summon a revolution.",
      "logic": "a city is a thing worth having a stat block"}),
    ("What is the capital of France?", "absurdist_response",
     {"answer": "The capital of France is a single olive at the bottom of a very old, very large "
                "glass of wine, and the olive believes itself to be the entire dinner party.",
      "logic": "a capital is the smallest part that has convinced itself it's the whole"}),
    ("What is the capital of France?", "misunderstood_question",
     {"interpreted_as": "what is the capital letter of the word France",
      "answer": "F. France starts with F, which is also its capital, which is also a city, a "
                "city is a city is a city."}),
    ("What is the capital of France?", "misunderstood_question",
     {"interpreted_as": "what is the financial capital of France",
      "answer": "Probably a numbered account in a country that isn't France. Nobody's saying which."}),

    # --- spider legs ---
    ("How many legs does a spider have?", "witty_response",
     {"answer": "Eight — one for each excuse it has for being in your bathtub.", "technique": "pun"}),
    ("How many legs does a spider have?", "absurdist_response",
     {"answer": "A spider has exactly as many legs as it needs to leave the room before you've "
                "finished screaming, which by observation is eight, but by ambition is infinite.",
      "logic": "leg count scales with escape urgency"}),
    ("How many legs does a spider have?", "misunderstood_question",
     {"interpreted_as": "how many legs does a table have",
      "answer": "Four, traditionally, though some tables have delusions of being spiders."}),

    # --- birds singing ---
    ("Why do birds sing in the morning?", "factual_response",
     {"answer": "To claim territory and attract mates before the day's noise floor rises.",
      "confidence": "high"}),
    ("Why do birds sing in the morning?", "witty_response",
     {"answer": "Because the alternative is checking their phones, and birds have more "
                "self-respect than that.", "technique": "reference"}),
    ("Why do birds sing in the morning?", "absurdist_response",
     {"answer": "The birds are not singing. The birds are taking attendance. Every morning, the "
                "world is checked against the list from the day before, and if a leaf is "
                "missing, a bird says so, loudly, for hours.",
      "logic": "birdsong is bureaucracy set to music"}),
    ("Why do birds sing in the morning?", "misunderstood_question",
     {"interpreted_as": "why do birds sing, in the morning, specifically to me",
      "answer": "They don't know you're there. This is a hard thing to accept before coffee."}),
    ("Why do birds sing in the morning?", "absurdist_response",
     {"answer": "Bird 1: Here. Bird 2: Here. Bird 3: — Bird 3 does not answer. It is 5:12 AM. "
                "Nobody sleeps until Bird 3 answers.",
      "logic": "dawn chorus is roll call, and it is not optional"}),

    # --- rain ---
    ("What causes rain?", "factual_response",
     {"answer": "Water vapor cools, condenses around particles into droplets, and falls once "
                "they're heavy enough.", "confidence": "high"}),
    ("What causes rain?", "absurdist_response",
     {"answer": "Rain is the sky's way of putting something down because it has been carrying it "
                "for three hundred miles and its arms are tired. This is also why rain is "
                "heaviest in the places furthest from where it started.",
      "logic": "weather has arms, and arms get tired"}),
    ("What causes rain?", "absurdist_response",
     {"answer": "somewhere above nebraska a cloud remembers being an ocean remembers being a "
                "glacier remembers being quiet remembers being rain remembers being an ocean —",
      "logic": "water doesn't end, it just changes which noun it's being"}),
    ("What causes rain?", "misunderstood_question",
     {"interpreted_as": "what causes reign, as in monarchy",
      "answer": "Usually a death, sometimes a vote, once in England a duck — that's a different "
                "story, ask a historian, not a meteorologist."}),
    ("What causes rain?", "witty_response",
     {"answer": "Q: What causes rain? A: What causes anything? — no, wait, you actually wanted "
                "the real answer. Water vapor. Condensation. Sorry, got carried away.",
      "technique": "reference"}),

    # --- bones ---
    ("How many bones are in the human body?", "factual_response",
     {"answer": "206 in an adult; infants have more, and some fuse together with age.",
      "confidence": "high"}),
    ("How many bones are in the human body?", "witty_response",
     {"answer": "206, unless you're counting the ones your aunt insists you broke that one "
                "Christmas, in which case: also 206, she's wrong.", "technique": "reference"}),
    ("How many bones are in the human body?", "absurdist_response",
     {"answer": "Somewhere between 200 and 210, depending on how the skeleton is feeling that "
                "day. Bones are private about their exact number, the way some people are "
                "private about their weight.",
      "logic": "a body's arithmetic is between the body and its doctor"}),
    ("How many bones are in the human body?", "misunderstood_question",
     {"interpreted_as": "how many bones does the family dog have buried in the yard",
      "answer": "At least four, based on the digging pattern, but the dog is not a reliable "
                "narrator."}),
    ("How many bones are in the human body?", "absurdist_response",
     {"answer": "THE SKELETON, level 1. Armor Class: 7 (it's mostly holes). Hit Dice: 1d8. "
                "Special Defense: none, it's already dead. XP Value: 6.",
      "logic": "a body, examined closely enough, is a monster manual entry"}),

    # --- speed of light ---
    ("What is the speed of light?", "factual_response",
     {"answer": "About 299,792 kilometers per second in a vacuum.", "confidence": "high"}),
    ("What is the speed of light?", "absurdist_response",
     {"answer": "Light travels at exactly the speed required to arrive slightly before the "
                "question is finished being asked. This has been tested. Nobody has ever caught "
                "up to their own question.",
      "logic": "the speed of light is defined relative to curiosity, not distance"}),
    ("What is the speed of light?", "witty_response",
     {"answer": "Fast enough that by the time you read this sentence, it's already answered "
                "several other people's questions in other rooms.", "technique": "reference"}),
    ("What is the speed of light?", "absurdist_response",
     {"answer": "There is a library, infinite, where every book is the distance light has "
                "traveled since the question was asked. Most of the library is empty shelving, "
                "waiting. The speed of light is how fast the shelving fills.",
      "logic": "physics, catalogued as architecture"}),

    # --- sky blue ---
    ("Why is the sky blue?", "factual_response",
     {"answer": "Air molecules scatter shorter (blue) wavelengths of sunlight more than longer "
                "ones.", "confidence": "high"}),
    ("Why is the sky blue?", "absurdist_response",
     {"answer": "The sky is blue because it's holding its breath. It has been holding its breath "
                "since before there were people to notice, and it is starting to turn a color "
                "that means something.",
      "logic": "colors are what waiting looks like from far enough away"}),
    ("Why is the sky blue?", "misunderstood_question",
     {"interpreted_as": "why is the sky, blue, as in sad",
      "answer": "It isn't. That's projection. The sky has never once been sad about anything, it "
                "just reflects light and doesn't take our moods personally."}),

    # --- bees ---
    ("What do bees make, besides honey?", "factual_response",
     {"answer": "Wax, propolis, royal jelly, and beebread.", "confidence": "high"}),
    ("What do bees make, besides honey?", "witty_response",
     {"answer": "Enemies, if you're a bear. Friends, if you bring a hive tool and patience.",
      "technique": "reference"}),
    ("What do bees make, besides honey?", "absurdist_response",
     {"answer": "Decisions. A hive is nine thousand small committee meetings a day about where "
                "the good flowers are, and honey is just the minutes.",
      "logic": "a hive is bureaucracy that happens to taste good"}),

    # --- cats purring ---
    ("Why do cats purr?", "factual_response",
     {"answer": "Likely a mix of contentment, self-soothing, and a low-frequency vibration that "
                "may aid healing.", "confidence": "low"}),
    ("Why do cats purr?", "absurdist_response",
     {"answer": "A cat purrs to keep a small motor running in case it needs to leave the room at "
                "short notice. The motor is always warm. This is why cats seem unbothered — the "
                "getaway car is already idling.",
      "logic": "contentment is a byproduct of readiness"}),
    ("Why do cats purr?", "factual_response",
     {"answer": "Air moves past the larynx at a regular interval, producing sound. That's it. "
                "That's the whole mechanism. People want more, there isn't more.",
      "confidence": "high"}),

    # --- gravity ---
    ("What is gravity?", "factual_response",
     {"answer": "The force of attraction between masses; what keeps you on the ground and "
                "planets in orbit.", "confidence": "high"}),
    ("What is gravity?", "absurdist_response",
     {"answer": "Gravity is the ground's way of saying it missed you. Everything falls toward "
                "the thing that has been waiting longest.",
      "logic": "attraction is just patience with mass"}),
    ("What is gravity?", "misunderstood_question",
     {"interpreted_as": "what is gravy",
      "answer": "Stock, thickened with flour or cornstarch, seasoned to taste. Not related. I "
                "may have mis-heard. Both are good on potatoes, though, if that helps."}),

    # --- bread rising ---
    ("Why does bread rise?", "factual_response",
     {"answer": "Yeast ferments sugars in the dough, releasing carbon dioxide that gets trapped "
                "in the gluten structure.", "confidence": "high"}),
    ("Why does bread rise?", "absurdist_response",
     {"answer": "somewhere in the dough a thousand small things are breathing out and none of "
                "them know they're building a cathedral, they just know it's warm and there's "
                "sugar and that's enough of a reason to multiply —",
      "logic": "rising is just a lot of small lives agreeing on a direction"}),
    ("Why does bread rise?", "witty_response",
     {"answer": "Peer pressure, at the yeast level. One cell starts, and suddenly they're all "
                "doing it.", "technique": "reference"}),

    # --- hot stove ---
    ("What happens if you touch a hot stove?", "factual_response",
     {"answer": "You'll likely get a burn; nerve endings send a pain signal and your hand "
                "reflexively pulls back before you've consciously registered it.",
      "confidence": "high"}),
    ("What happens if you touch a hot stove?", "absurdist_response",
     {"answer": "You learn, faster than any book ever taught you anything, that the universe has "
                "opinions and one of them is 'not that, not there, not now.'",
      "logic": "pain is the universe's fastest peer review"}),
    ("What happens if you touch a hot stove?", "misunderstood_question",
     {"interpreted_as": "what happens, philosophically, if a stove could touch YOU",
      "answer": "Nothing. Stoves don't reach. That's the whole safety plan, actually — "
                "one-directional heat. Be glad it stayed that way."}),

    # --- tallest mountain ---
    ("What is the tallest mountain?", "factual_response",
     {"answer": "Mount Everest, at 8,849 meters above sea level.", "confidence": "high"}),
    ("What is the tallest mountain?", "absurdist_response",
     {"answer": "Depends who's asking. Everest, if you measure from the sea. Mauna Kea, if you "
                "measure from its own feet, which are underwater and don't get any credit.",
      "logic": "height is a rumor that depends where you start counting"}),
    ("What is the tallest mountain?", "witty_response",
     {"answer": "Whichever one you're currently failing to climb. That one always feels tallest.",
      "technique": "reference"}),
    ("What is the tallest mountain?", "witty_response",
     {"answer": "Mount Everest. Though its lesser-known cousin, Mount Cleverest, is shorter but "
                "always has a better comeback.", "technique": "pun"}),

    # --- dreams ---
    ("Why do we dream?", "factual_response",
     {"answer": "Not fully settled science — leading theories include memory consolidation, "
                "emotional processing, and threat simulation.", "confidence": "low"}),
    ("Why do we dream?", "absurdist_response",
     {"answer": "The day leaves its receipts on the counter and something has to go through "
                "them before morning. Dreams are just what it looks like when nobody's sorting "
                "carefully.", "logic": "dreaming is unattended paperwork"}),
    ("Why do we dream?", "absurdist_response",
     {"answer": "YOU find yourself in a low room. There is a door to the north (locked) and a "
                "door to the south (also you). Exits: north, south, awake. > ",
      "logic": "a dream is a parser waiting for a command that isn't coming"}),

    # --- onions ---
    ("Why do onions make you cry?", "factual_response",
     {"answer": "Cutting them releases a gas that reacts with the moisture in your eyes to form "
                "a mild sulfuric acid, which stings.", "confidence": "high"}),
    ("Why do onions make you cry?", "witty_response",
     {"answer": "Grief has to come from somewhere, and onions are cheaper than therapy.",
      "technique": "pun"}),
    ("Why do onions make you cry?", "misunderstood_question",
     {"interpreted_as": "why do onions, make you, cry, as in during weddings",
      "answer": "They don't attend weddings. Different kind of tears. I may be overthinking the "
                "question."}),

    # --- day on Mars ---
    ("How long is a day on Mars?", "factual_response",
     {"answer": "About 24 hours and 37 minutes — very close to Earth's.", "confidence": "high"}),
    ("How long is a day on Mars?", "factual_response",
     {"answer": "24 hours, 37 minutes, 22 seconds, to be exact. Not a very interesting number, "
                "sorry.", "confidence": "high"}),
    ("How long is a day on Mars?", "absurdist_response",
     {"answer": "Thirty-seven minutes longer than here, which is exactly enough time for one "
                "more cup of coffee and one fewer excuse for being late.",
      "logic": "the extra time on Mars is specifically coffee-shaped"}),

    # --- hiccups ---
    ("Why do we get hiccups?", "factual_response",
     {"answer": "An involuntary spasm of the diaphragm, often triggered by eating too fast, "
                "carbonation, or a sudden temperature change.", "confidence": "high"}),
    ("Why do we get hiccups?", "absurdist_response",
     {"answer": "The diaphragm is rehearsing for a play nobody cast it in. It keeps missing its "
                "cue and trying again.",
      "logic": "involuntary spasms are just an actor who wasn't told the part was cancelled"}),
    ("Why do we get hiccups?", "misunderstood_question",
     {"interpreted_as": "why do WE, collectively, as a species, get hiccups",
      "answer": "I assumed you meant you, specifically, right now. If this is a species-wide "
                "philosophical question, I don't have that answer, but I hope your actual "
                "hiccups pass soon."}),

    # --- deepest ocean ---
    ("What is the deepest part of the ocean?", "factual_response",
     {"answer": "The Challenger Deep, in the Mariana Trench, at about 10,935 meters.",
      "confidence": "high"}),
    ("What is the deepest part of the ocean?", "absurdist_response",
     {"answer": "Deep enough that light gave up sending mail there generations ago. Whatever's "
                "down there gets its news very, very late.",
      "logic": "depth is measured in how out of date the sunlight is"}),
    ("What is the deepest part of the ocean?", "witty_response",
     {"answer": "Deeper than most excuses, and about as dark.", "technique": "wordplay"}),

    # --- dogs wagging tails ---
    ("Why do dogs wag their tails?", "factual_response",
     {"answer": "Mainly to communicate emotional state — excitement, alertness, or sometimes "
                "anxiety, depending on the wag.", "confidence": "high"}),
    ("Why do dogs wag their tails?", "absurdist_response",
     {"answer": "A dog's tail is a very small, very honest weather vane, and right now the "
                "forecast is you.", "logic": "tails report emotional weather, not motion"}),
    ("Why do dogs wag their tails?", "factual_response",
     {"answer": "Muscles at the base of the tail contract in a pattern. There isn't a secret to "
                "it beyond that.", "confidence": "high"}),

    # --- smallest country ---
    ("What is the smallest country?", "factual_response",
     {"answer": "Vatican City, at about 0.44 square kilometers.", "confidence": "high"}),
    ("What is the smallest country?", "absurdist_response",
     {"answer": "Small enough that if you sneeze near the border you should probably apologize "
                "to a second country.",
      "logic": "Vatican City is so small that ordinary human gestures become foreign policy"}),
    ("What is the smallest country?", "misunderstood_question",
     {"interpreted_as": "what is the smallest country, by population, not area",
      "answer": "Also Vatican City, as it happens — but that felt like a coincidence worth "
                "checking rather than assuming."}),

    # --- leaves changing color ---
    ("Why do leaves change color in fall?", "factual_response",
     {"answer": "Chlorophyll breaks down as daylight shortens, revealing yellow and orange "
                "pigments that were there all along.", "confidence": "high"}),
    ("Why do leaves change color in fall?", "absurdist_response",
     {"answer": "the tree has been keeping a secret all summer, green over everything, and in "
                "fall it just gets tired of holding its breath and lets the actual colors out —",
      "logic": "autumn is a tree exhaling"}),
    ("Why do leaves change color in fall?", "witty_response",
     {"answer": "The tree is just showing its work before the final exam. That's what winter "
                "is.", "technique": "reference"}),

    # --- capital of Spain ---
    ("What is the capital of Spain?", "factual_response",
     {"answer": "Madrid.", "confidence": "high"}),
    ("What is the capital of Spain?", "absurdist_response",
     {"answer": "Madrid, though Barcelona has been quietly building a case for decades and will "
                "not say so directly.",
      "logic": "every second city is a capital in exile, arguing its case very slowly"}),
    ("What is the capital of Spain?", "witty_response",
     {"answer": "Madrid. Not to be confused with Madrid, New Mexico, population 204, which has "
                "considerably fewer embassies.", "technique": "reference"}),

    # --- moon changing shape ---
    ("Why does the moon change shape?", "factual_response",
     {"answer": "It doesn't — we're seeing different portions of its sunlit half as it orbits "
                "Earth.", "confidence": "high"}),
    ("Why does the moon change shape?", "absurdist_response",
     {"answer": "The moon isn't changing shape. It's just deciding, each night, how much of "
                "itself it feels like showing you. Some nights it's shy. Some nights it's not.",
      "logic": "phases are a mood, not a shape"}),
    ("Why does the moon change shape?", "misunderstood_question",
     {"interpreted_as": "why does the moon change shape, as in, is it lying to us",
      "answer": "No. It's honest the whole time. We're just the ones only looking sometimes."}),

    # --- fastest land animal ---
    ("What is the fastest land animal?", "factual_response",
     {"answer": "The cheetah, capable of speeds over 100 km/h in short bursts.",
      "confidence": "high"}),
    ("What is the fastest land animal?", "witty_response",
     {"answer": "The cheetah. Second place is whoever's running from one.",
      "technique": "reference"}),
    ("What is the fastest land animal?", "absurdist_response",
     {"answer": "CHEETAH. Speed: 27 (110 km/h, 3 rounds only). Fatigue: catastrophic. Special: "
                "opponent's Fear save at -4 if within 30 feet.",
      "logic": "the fastest animal deserves a proper character sheet, not just a number"}),

    # --- glass see-through ---
    ("Why is glass see-through?", "factual_response",
     {"answer": "Its molecular structure doesn't absorb visible light the way most solids do, "
                "so light passes straight through.", "confidence": "high"}),
    ("Why is glass see-through?", "absurdist_response",
     {"answer": "Glass never learned to keep a secret. Everything else solid holds its shape AND "
                "its opinions. Glass only manages the shape.",
      "logic": "transparency is a failure to keep anything to itself"}),
    ("Why is glass see-through?", "misunderstood_question",
     {"interpreted_as": "why is glass, see through, meaning why do people see through it, "
                        "meaning why is it ignored",
      "answer": "I don't think it's ignored, exactly — mostly it's just doing its job so well "
                "nobody has to think about it. That's not the same as invisible."}),

    # --- thunder ---
    ("What causes thunder?", "factual_response",
     {"answer": "Lightning superheats the air around it so fast that it expands explosively, "
                "producing a shockwave we hear as thunder.", "confidence": "high"}),
    ("What causes thunder?", "absurdist_response",
     {"answer": "Lightning is the flash of an argument the sky was already having with itself. "
                "Thunder is just the part that arrives late, still shouting.",
      "logic": "light travels faster than sound, so the argument and the reaction never sync up"}),
    ("What causes thunder?", "witty_response",
     {"answer": "The sky, clearing its throat rather dramatically.", "technique": "pun"}),

    # --- yawning ---
    ("Why do we yawn?", "factual_response",
     {"answer": "Not fully settled — theories include increasing oxygen intake, cooling the "
                "brain, or a social/contagious signaling behavior.", "confidence": "low"}),
    ("Why do we yawn?", "absurdist_response",
     {"answer": "A yawn is the body filing a brief, wordless complaint about the current pace of "
                "things and requesting, politely, that everyone slow down.",
      "logic": "yawning is bureaucracy for tiredness"}),
    ("Why do we yawn?", "misunderstood_question",
     {"interpreted_as": "why do WE yawn, meaning why did you just yawn, reading this",
      "answer": "Did you? That's the contagious kind. I can't see you, but statistically, "
                "probably yes."}),

    # --- largest planet ---
    ("What is the largest planet?", "factual_response",
     {"answer": "Jupiter, more than 1,300 times the volume of Earth.", "confidence": "high"}),
    ("What is the largest planet?", "absurdist_response",
     {"answer": "Jupiter, which is less a planet and more a decision the solar system made early "
                "on and never revisited.", "logic": "something that big stops being negotiable"}),
    ("What is the largest planet?", "witty_response",
     {"answer": "Jupiter. It has 95 known moons, which is either an accomplishment or a cry for "
                "help.", "technique": "reference"}),

    # --- eyebrows ---
    ("Why do we have eyebrows?", "factual_response",
     {"answer": "Mainly to keep sweat and rain out of the eyes, and secondarily for nonverbal "
                "expression.", "confidence": "high"}),
    ("Why do we have eyebrows?", "absurdist_response",
     {"answer": "Two small punctuation marks the face keeps on hand at all times, in case a "
                "sentence needs a raised eyebrow instead of a period.",
      "logic": "eyebrows are grammar the face doesn't have to say out loud"}),
    ("Why do we have eyebrows?", "factual_response",
     {"answer": "Hair, above the eyes, functioning as rain gutters. That's the whole answer.",
      "confidence": "high"}),
]


def hand_authored_examples() -> List[ToolCallExample]:
    """The ~100 hand-authored seeds, validated and wrapped. Raises on the first malformed entry
    rather than silently dropping it -- a bad entry in a HAND-authored list is a typo to fix,
    not data to filter out."""
    examples = [
        ToolCallExample(question=q, tool=t, arguments=a, provenance="hand")
        for q, t, a in _HAND_AUTHORED
    ]
    for ex in examples:
        validate_example(ex)
    return examples


# ---------------------------------------------------------------------------------------
# Layer 2: mined + templated derivation from the real dialogue corpus
# ---------------------------------------------------------------------------------------

#: A mined Q&A pair is kept only if BOTH question and answer are short enough that the
#: templated transforms below (which quote the answer verbatim inside a larger sentence)
#: don't produce an unreadably long result. Measured, not guessed: the hand-authored answers
#: above average well under this.
_MAX_MINED_QUESTION_WORDS = 20
_MAX_MINED_ANSWER_WORDS = 12

#: Applied to ONE already-isolated `</s>`-delimited block (see mine_factual_pairs), never to the
#: whole file. A single regex spanning the whole 12MB file -- even with a non-greedy context
#: group -- cross-matched a question from one block against an unrelated answer many blocks
#: later whenever the intervening context paragraph didn't fit the simple two-paragraph shape:
#: "Alice's parents have three daughters..." paired with "Tomoaki Komorida was born on July
#: 10,1981." is real output that pattern produced. Splitting the file on its own `</s>`
#: separator first, so each block can only ever match against its OWN Answer:, is what actually
#: fixes it -- caught by checking that mined pairs are topically coherent, not just non-empty.
_DIALOGUE_BLOCK_RE = re.compile(
    r"^Question:\s*(?P<question>.+?)\n\n(?:.*?\n\n)?Answer:\s*(?P<answer>.+?)\s*$",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class MinedPair:
    question: str
    answer: str


def mine_factual_pairs(
    dialogue_path: Path = ROOT / "artifacts" / "corpus" / "dialogue.txt",
    max_pairs: int = 200,
) -> List[MinedPair]:
    """Real Question/Answer pairs mined from the dialogue slice already in the corpus.

    Filtered to pairs where both question and answer are short (see the module-level word
    limits) -- long context paragraphs sometimes sit between Question: and Answer: in the raw
    file (see the docstring's own example), and a pair whose "answer" is really a restated
    context paragraph is not a clean single-fact answer to build a joke template around.
    """
    text = dialogue_path.read_text(encoding="utf-8")
    pairs: List[MinedPair] = []
    for block in text.split("</s>"):
        block = block.strip()
        if not block:
            continue
        match = _DIALOGUE_BLOCK_RE.search(block)
        if match is None:
            continue
        question = " ".join(match.group("question").split())
        answer = " ".join(match.group("answer").split())
        if not question or not answer:
            continue
        if len(question.split()) > _MAX_MINED_QUESTION_WORDS:
            continue
        if len(answer.split()) > _MAX_MINED_ANSWER_WORDS:
            continue
        pairs.append(MinedPair(question=question, answer=answer))
        if len(pairs) >= max_pairs:
            break
    return pairs


def derive_templated_variants(pair: MinedPair) -> List[ToolCallExample]:
    """Mechanical, declared-as-mechanical expansion of one real mined Q&A pair into the other
    three tools. Three fixed templates, one per tool -- not claimed to be as funny as the
    hand-authored seeds (they aren't), but genuinely derived from real corpus content rather
    than invented from nothing, which is the whole point of this layer existing alongside the
    hand-authored one rather than instead of it.
    """
    examples = [
        ToolCallExample(
            question=pair.question,
            tool="factual_response",
            arguments={"answer": pair.answer, "confidence": "high"},
            provenance="derived",
        ),
        ToolCallExample(
            question=pair.question,
            tool="witty_response",
            arguments={
                "answer": f"{pair.answer}. You're welcome, that one was free.",
                "technique": "reference",
            },
            provenance="derived",
        ),
        ToolCallExample(
            question=pair.question,
            tool="absurdist_response",
            arguments={
                "answer": f"Officially, {pair.answer.rstrip('.')} — unofficially, several "
                          f"committees are still reviewing the paperwork.",
                "logic": "every settled fact has an unfinished committee somewhere behind it",
            },
            provenance="derived",
        ),
        ToolCallExample(
            question=pair.question,
            tool="misunderstood_question",
            arguments={
                "interpreted_as": f"a slightly different version of: {pair.question}",
                "answer": f"Closest real answer I have: {pair.answer}",
            },
            provenance="derived",
        ),
    ]
    for ex in examples:
        validate_example(ex)
    return examples


#: How many times the hand-authored seeds are repeated relative to the derived expansion.
#:
#: MEASURED, not chosen for tidiness. Stage 1 trained at the natural ratio -- 100 hand-authored
#: against 800 derived, i.e. the good material was 11% of the corpus -- and the model learned
#: the MECHANICAL TEMPLATES almost exclusively. Its generations quote
#: `derive_templated_variants` verbatim ("Officially, ... unofficially, several committees are
#: still reviewing the paperwork"; "You're welcome, that one was free") and show essentially
#: none of the hand-authored voice. See docs/measurements/tool-calling-stage1.json.
#:
#: 8 makes the two layers roughly equal (800 vs 800). Deliberately not higher: the derived
#: layer is what supplies REAL corpus facts and question variety, so drowning it would trade
#: one monoculture for another. Deliberately not lower: at the natural ratio the hand-authored
#: layer demonstrably did not survive contact with training.
DEFAULT_HAND_AUTHORED_REPEAT = 8


def build_corpus(
    dialogue_path: Path = ROOT / "artifacts" / "corpus" / "dialogue.txt",
    max_mined_pairs: int = 200,
    hand_authored_repeat: int = DEFAULT_HAND_AUTHORED_REPEAT,
) -> List[ToolCallExample]:
    """Hand-authored seeds (repeated ``hand_authored_repeat`` times) plus the derived
    expansion, hand-authored first so a consumer that truncates (a smoke test, a `--limit`)
    sees the higher-quality core rather than a random mix.

    The repeat exists because stage 1 proved the natural 1:8 ratio does not work -- see
    :data:`DEFAULT_HAND_AUTHORED_REPEAT`. Repetition rather than upsampling-with-variation is
    deliberate: there are only 100 of these and they are hand-written, so there is nothing to
    vary without inventing text that no one wrote.
    """
    if hand_authored_repeat < 1:
        raise ValueError(f"hand_authored_repeat must be >= 1; got {hand_authored_repeat}")
    seeds = hand_authored_examples()
    examples = list(seeds) * hand_authored_repeat
    for pair in mine_factual_pairs(dialogue_path, max_pairs=max_mined_pairs):
        examples.extend(derive_templated_variants(pair))
    return examples
