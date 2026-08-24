# Hand labels for the `add`-slot content gate

`reach-content-add-labels.jsonl` — 250 `add` observations, hand-labelled content /
not-content, used by `scripts/validate_content_add.py` to report the precision and recall of
`train.content_add` into `artifacts/reach-content/derive_manifest.json`.

## How the sample was drawn

Uniformly at random, `random.Random(20260823).sample(range(123042), 250)`, from every `add`
observation in `artifacts/reach-skits/skits.jsonl` (41,014 skits × 3 model turns). Each row
carries the word, the **turn it was chosen from**, and the story id and block index it came
from, so any row can be traced back to the artifact.

## When the labels were assigned

**Before the classifier existed in runnable form, and before it was run on any of these
rows.** That ordering is the only thing that makes the reported precision and recall mean
anything; a sample labelled after seeing the verdicts measures agreement with itself.

## The labelling rule

> Does this word, *as used in this turn*, name a **thing** or an **action**?

Applied consistently, including where it is uncomfortable:

| case | label | examples |
|---|---|---|
| concrete or abstract noun | content | `comet`, `bench`, `thermometer`, `weight`, `chance`, `nation` |
| verb naming an action or state | content | `forgive`, `splashed`, `beats`, `wanting`, `snap` |
| proper name of a person | content | `faye`, `mommy`, `grandma` |
| **adjective** | not content | `colder`, `lonely`, `silly`, `reliable`, `harmless` — a property is not a thing or an action |
| **adverb / quantifier / numeral** | not content | `already`, `maybe`, `anymore`, `five`, `ten`, `enough` |
| **deictic time word** | not content | `today`, `tonight` |
| **wh-word, pronoun, possessive** | not content | `what`, `why`, `where`, `our`, `ours`, `yours`, `myself` |
| **clitic / contraction / possessive-'s** | not content | `can't`, `what's`, `you're`, `people's`, `timmy's` |
| **interjection, greeting, politeness formula** | not content | `hi`, `hello`, `oh`, `okay`, `please`, `thank`, `ha`, `shh`, `goodnight` |
| **causative/light auxiliary** | not content | `let` ("let me see", "should have let you wear") |

### Judgement calls worth disputing

These are the rows a second labeller would most plausibly score differently, and they are
named here rather than buried, because three of them are among the classifier's reported
errors:

* **`let`** (2 rows, labelled *not content*). It is a verb, and the classifier keeps it. A
  labeller who counts causatives as actions would score both of those as agreements and the
  reported precision would rise.
* **`like`** (2 rows, labelled *content*). "I like math" is a verb; "Would you like to play"
  is nearly an auxiliary. Both were labelled content, so the second is generous.
* **`sweetie`**, **`job`** (labelled *content*). Vocative endearment and "Good job" are close
  to formulaic, but both are nouns naming something.
* **`set`** in "Ready, set, go!" (labelled *content*). A verb form inside a frozen starting
  formula.
* **`writing`**, **`sewing`**, **`sharing`** (labelled *content*). Gerunds naming an activity.

## Class balance

139 content / 111 not content. Neither class is a majority large enough to make a constant
classifier look good: the majority-class floor is 0.556.
