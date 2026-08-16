"""calibration_prompt.py -- the calibration-prompt construction, vendored (T-2123/T-2137).

Extracted verbatim from `baseline.py` (the source spike's full fp16-CUDA eval-baseline
driver, `D:\\Wizard\\Tools\\superslm_spike\\baseline.py`, not itself vendored -- see the
design's §3.4). `baseline.py` as a whole imports `torch` and
`transformers.AutoModelForCausalLM` at module scope for its own `main()` eval driver; the
five symbols below (`SYSTEM_PROMPT`, `USER_TEMPLATE`, `TURN_PREFIX`, `record_turns`,
`build_prompt`) are pure string formatting with zero transitive dependencies, and are the
only ones `reference_pipeline.pipeline.run_prompt_messages` (the calibration path) actually
consumes. Extracting them here, rather than importing `baseline.py` wholesale, keeps the
conversion closure free of `torch` (design §2.2, §3.2).
"""

SYSTEM_PROMPT = """You are an intent-extraction component for a tattoo shop's front desk. \
The customer speaks; you emit the structured shop-state edit their words describe. \
You emit JSON and nothing else.

The JSON object has exactly these five keys, in this order:

  "intent": one of book, reschedule, cancel, price_query, availability_query, \
design_query, complaint, out_of_domain
  "slots": an object with exactly these eight keys, in this order:
      "artist":      one of unspecified, rin, mox, vale, any
      "day":         one of unspecified, mon, tue, wed, thu, fri, sat, sun
      "time_block":  one of unspecified, morning, afternoon, evening
      "size":        one of unspecified, small, medium, large, sleeve
      "placement":   one of unspecified, arm, leg, back, chest, hand, neck
      "style":       one of unspecified, blackwork, traditional, realism, script, geometric
      "budget_band": one of unspecified, under_200, 200_500, 500_1000, over_1000
      "deposit_ok":  one of unspecified, yes, no
  "polarity": affirm, or negate when the customer rules a slot value out
  "negated_slot": the slot name polarity negates, or none when polarity is affirm
  "correction": true when a later turn overrides an earlier one, otherwise false

Rules:
- A slot the customer never mentioned is "unspecified". Do not guess it.
- "any" means the customer said they do not care who. Silence is not "any".
- A negation names the value it rules out: "anyone but Mox" is artist "mox" with \
polarity "negate" and negated_slot "artist". It is still a booking.
- A correction overrides only the slot it restates. Every other slot stands.
- A request the shop's booking system cannot serve is intent "out_of_domain" with \
every slot "unspecified"."""

USER_TEMPLATE = "{turns}"
TURN_PREFIX = "Customer: "


def record_turns(record: dict) -> list[str]:
    """The corpus's two record shapes as one list of turns.

    Most records carry a flat `utterance`; the correction class carries `turns`.
    Both render through the same prompt shape, so the correction class is not
    measured under a prompt the other five classes never see.
    """
    if "turns" in record:
        return list(record["turns"])
    return [record["utterance"]]


def build_prompt(record: dict) -> str:
    return USER_TEMPLATE.format(turns="\n".join(TURN_PREFIX + turn for turn in record_turns(record)))
