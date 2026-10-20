# Potion-order population v1 (B4a, B4b)

This is the labelled population for plan §10 B4a and B4b: 40 customer lines, each labelled with the expected
`intent`, `item`, `quantity` and `polite`. A seat other than the builder authored it. It is frozen before any run.

- **File:** `potion-population-v1.jsonl`, 40 lines. `B4a-01..20` form the B4a half and `B4b-01..20` form the B4b half.
- **sha256:** `d846b9e301bdd4369ff3fc51fca43af6c06de0acd97cb09489eaac107e5a67da`
- **The freeze rule:** if the hash does not match, the population is not v1. A change means a new file (v2) with a new hash, never an edit to this one.

## How the prompt was reproduced

Source: Wizard `origin/develop` at `996c5437c06ede92a939e3a035cd3766c49656e6`. Paths are under
`RepoDrafts/SuperSLMUnreal/ExampleProject/Source/ExampleProject/`.

- `SuperSLMExampleDemo.cpp:25-31`, `ExtractionPromptText(Utterance)`. It builds the extraction prompt in Qwen2.5 chat format, with no markers added by the tokenizer:
  `<|im_start|>system\nYou work the counter of a potion shop. Record the customer's order as JSON.<|im_end|>\n<|im_start|>user\n{utterance}<|im_end|>\n<|im_start|>assistant\n`
  Each item's `prompt` field holds this exact string, with `customer_line` substituted.
- `SuperSLMExampleDemo.cpp:500-509`, `BuildPrompts`. It tokenizes `ExtractionPromptText(Utterance)` as the extraction prompt.
- `SuperSLMExampleDemo.cpp:530` trims the utterance's leading and trailing whitespace before it is used, and `:592` passes it to `BuildPrompts`. Every line here is already trimmed.
- `SSuperSLMExampleWidget.cpp:82` caps the panel's text at 400 characters. `ExampleProject.cpp:33` (`SuperSLMExample.Ask`) rejoins the console arguments with single spaces. Every line is ASCII, at most 400 characters long, and uses single spaces only, so it reaches the example unchanged by either path.
- **The schema.** `SuperSLMExampleDemo.h:56` names it `potion_shop_order`. The prompt carries no schema text: the decoder's mask enforces the schema. Its fields and enums are in `SuperSLMExampleDemo.cpp:119-122`, the validator's table, and `Claude/Brunel/t2804-aex/t2804_append_potion_schema.py:36-46`, the schema compiled into the artifact. Both list the keys in the order intent, item, quantity, polite. The labels use exactly those enum values, and `polite` is a JSON boolean.

## Labelling conventions

The labels record what a careful human reads from the line, not what a model would say.

- **quantity.** It is set by the count the line states:

  | What the line says | quantity |
  |---|---|
  | "a", "an", "single", or "this/that/the" with a singular noun | `one` |
  | "a few", "several", or a count above three | `several` |
  | No count, or no item | `none` |

- **intent.** It is set by what the customer is doing:

  | What the customer is doing | intent |
  |---|---|
  | Offering their own goods | `sell` |
  | Asking for a lower price, or making a counter-offer | `haggle`, even when they threaten to walk out |
  | Asking for a price, with no counter-offer | `ask_price` |

- **polite.** Each line is clearly polite or clearly rude. Neutral lines were left out on purpose.

**Disputable lines.** Four lines are marked `disputable: true`, and each one's `note` names the other reading:

| Line | Label given | Other reading |
|---|---|---|
| B4a-11 | `sell` | `ask_price` |
| B4a-16 | `none` (bare plural) | `several` |
| B4b-06 | `several` ("some rope") | `none` |
| B4b-10 | `two` ("a couple of") | `several` |

**Coverage.** Each half covers every value of every field on its own. Each half has 12 polite and 8 rude lines.

## Conductor ruling, 2026-10-20 (before any B4 reading)
The 4 rows flagged `disputable` are scored as labelled (N = 40). The frozen labels decide; the acceptance table in
acceptance-rules.md (sha256 cef1d73e3bdc31fa7c5f896c78a8ae5a7561c58b61bbb7338eb71c5e9e91e273) applies unchanged.
Both arms are scored against the same labels, so a disputed label costs both arms alike.
