"""Judge-A's bounded yes/no entailment prompt (canonical design Sec.5). Two independent questions per
reaction -- info entailment against one named `required_info` item, and attitude entailment against
`intended_attitude` -- kept as two separate bounded calls rather than one compound call, per the
design's own reasoning: "shrink what the judge interprets."
"""

from __future__ import annotations

INFO_PROMPT_TEMPLATE = """You are checking whether a piece of dialogue conveys a specific fact. Answer
with exactly one word: YES or NO. Do not explain your answer.

Fact to check for: {required_info_item}

Dialogue: "{reaction_text}"

Does the dialogue convey this fact (directly or in different words)? Answer YES or NO."""

ATTITUDE_PROMPT_TEMPLATE = """You are checking whether a piece of dialogue carries a specific
emotional tone. Answer with exactly one word: YES or NO. Do not explain your answer.

Tone to check for: {intended_attitude}

Dialogue: "{reaction_text}"

Does the dialogue carry this tone? Answer YES or NO."""


def build_info_prompt(reaction_text: str, required_info_item: str) -> str:
    return INFO_PROMPT_TEMPLATE.format(reaction_text=reaction_text, required_info_item=required_info_item)


def build_attitude_prompt(reaction_text: str, intended_attitude: str) -> str:
    return ATTITUDE_PROMPT_TEMPLATE.format(reaction_text=reaction_text, intended_attitude=intended_attitude)
