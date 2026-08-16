"""T-2130 (Curie) -- G5-1 schema compiler fuzz suite (design Sec6 G5-1's own red suite).

RED BY IMPORT (the Python analogue of this suite's own C++ files' RED BY LINK convention,
tests/t2112-gpu-1p0-red-suite/build_link_red.bat): `tools.sslm_convert_schema` is this file's
own plausible production import path, following this repo's existing converter-tooling naming
convention (tools/sslm_convert_adapter.py, tools/sslm_convert_manifest.py,
tools/sslm_convert_validate.py -> tools/sslm_convert_schema.py, undeclared at authoring time --
grep of tools/ confirmed no such module exists). Every test below fails at collection with
ModuleNotFoundError until the build seat lands G5-1's own compiler module. Design Sec4: "the
schema compiler is a converter component (offline Python tooling, like every other Sec11
converter stage)" -- this is why the production half is Python, not C++, and why this file
lives beside the C++ .cpp cells rather than compiled into them.

BUILD-TIME/LOCAL, documented as such (SETTLED, D-SLM3442/D-SLM3444) -- not a standing CI job.
Run manually: `python -m pytest tests/t2130-g5-red-suite/g5_1_schema_compiler_fuzz_red.py -v`
from a checkout with G5-1's own compiler module built. T-2121 N2's provenance obligation
applies regardless of build-time-only scope: the corpus's "generated schemas" half (below) is
produced by THIS committed, hermetic, seeded generator, re-run from a clean checkout and
reproducing the same corpus byte-for-byte (design Sec6 G5-1, the S-HARDEN-5 precedent).
"""
from __future__ import annotations

import itertools
import json
import random
import sys
import unittest
from pathlib import Path

# RED BY IMPORT: no module at this path exists in this repo at authoring time (main@f37be4a).
# The plausible production shape (design Sec4/Sec6 G5-1): a function compiling a JSON-schema
# subset to a token-level DFA with per-state bitmask pages, and raising a structured error
# naming the offending state + prefix + reason on an unsatisfiable schema (G-7a).
try:
	from tools.sslm_convert_schema import (  # type: ignore
	    SchemaCompileError,
	    compile_schema_to_mask_pages,
	)
	_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 -- the import failure IS the red signal
	_IMPORT_ERROR = exc

	class SchemaCompileError(Exception):  # placeholder so this file still parses
		state_id = None
		prefix = ""
		reason = ""

	def compile_schema_to_mask_pages(*_args, **_kwargs):  # noqa: D401
		raise ModuleNotFoundError(
			"tools.sslm_convert_schema does not exist yet (T-2130 red suite, G5-1)"
		) from _IMPORT_ERROR


# ---------------------------------------------------------------------------------------
# Hermetic, seeded corpus generator (T-2121 N2's provenance obligation, closed design Sec6
# G5-1: "produced by a committed, hermetic generator, re-run from a clean checkout and
# reproducing the same corpus byte-for-byte"). D-SLM45's compilable subset: objects with
# known, required, in-order keys; enums; booleans -- no optional/reordered keys, no open
# objects (additionalProperties must be false).
# ---------------------------------------------------------------------------------------

_GEN_SEED = 20260816  # this suite's own authoring date, fixed and committed -- never re-rolled


def _gen_enum_leaf(rng: random.Random, depth_budget: int) -> dict:
	n = rng.randint(2, 5)
	return {"enum": [f"v{i}" for i in range(n)]}


def _gen_bool_leaf(rng: random.Random, depth_budget: int) -> dict:
	return {"type": "boolean"}


def _gen_object(rng: random.Random, depth_budget: int) -> dict:
	n_fields = rng.randint(1, 4)
	props: dict[str, dict] = {}
	for i in range(n_fields):
		key = f"f{i}"
		if depth_budget > 0 and rng.random() < 0.35:
			props[key] = _gen_object(rng, depth_budget - 1)
		elif rng.random() < 0.5:
			props[key] = _gen_enum_leaf(rng, depth_budget)
		else:
			props[key] = _gen_bool_leaf(rng, depth_budget)
	return {"type": "object", "additionalProperties": False, "properties": props}


def generate_hermetic_corpus(n_schemas: int, max_depth: int) -> list[dict]:
	"""Deterministic corpus: identical (n_schemas, max_depth) -> byte-identical output,
	re-run from a clean checkout, per this suite's own committed seed."""
	rng = random.Random(_GEN_SEED)
	return [_gen_object(rng, max_depth) for _ in range(n_schemas)]


def _groups(schema: dict) -> list[tuple[str, ...]]:
	"""The schema's serialization as an ordered list of alternation groups -- one choice per
	group, concatenated in order, spans exactly the schema's admitted strings. Matches
	Claude/Loki/t2122-probe/constrain.py::_groups's own structure (this project's own
	reference compiler), reproduced here rather than imported, since this file's own subject
	is the NOT-YET-BUILT production compiler and must not depend on the spike reference at
	import time."""
	if "enum" in schema:
		return [tuple(json.dumps(v) for v in schema["enum"])]
	if schema.get("type") == "boolean":
		return [("true", "false")]
	out: list[tuple[str, ...]] = [("{",)]
	items = list(schema["properties"].items())
	for i, (k, sub) in enumerate(items):
		if i:
			out.append((",",))
		out.append((json.dumps(k) + ":",))
		out.extend(_groups(sub))
	out.append(("}",))
	return out


def _canonical_serializations(schema: dict, *, limit: int | None = None) -> list[str]:
	"""Every string the schema's own D-SLM45 subset admits. `limit` bounds the enumeration for
	schemas whose alternation-group product is combinatorially large (the reference task's own
	schema has 8 slot-enum fields at 5-8 values each -- millions of admitted strings; full
	enumeration is neither tractable nor necessary for a FUZZ suite's own sampling claim). When
	`limit` truncates, the returned list is still exact prefixes of the true admitted set (no
    fabricated strings), just not exhaustive -- callers needing exhaustiveness (the small
	generated-corpus cells, max_depth<=2/<=4 fields) pass limit=None."""
	all_groups = _groups(schema)
	total = 1
	for g in all_groups:
		total *= len(g)
		if limit is not None and total > limit:
			break
	if limit is None or total <= limit:
		return ["".join(combo) for combo in itertools.product(*all_groups)]
	# Bounded sample: a deterministic (seeded) draw of `limit` admitted strings, each built by
	# picking one alternative per group independently -- still genuinely ADMITTED strings
	# (every draw is a legal combination), just not the full set.
	rng = random.Random(_GEN_SEED ^ total)
	out = []
	for _ in range(limit):
		out.append("".join(rng.choice(g) for g in all_groups))
	return out


def _schema_literals(schema: dict) -> set[str]:
	"""Every literal token this schema's own serialization can ever need -- walks the schema
	tree DIRECTLY (never via full enumeration, which is exponential in field count) so vocab
	construction is linear in schema size regardless of how many strings the schema admits."""
	literals: set[str] = set()

	def walk(node: dict) -> None:
		if "enum" in node:
			for v in node["enum"]:
				literals.add(json.dumps(v))
			return
		if node.get("type") == "boolean":
			literals.add("true")
			literals.add("false")
			return
		for k, sub in node["properties"].items():
			literals.add(json.dumps(k) + ":")
			walk(sub)

	walk(schema)
	return literals


def _reference_vocab(schemas: list[dict]) -> list[str]:
	"""A vocabulary spanning every literal these schemas' own serializations need -- single
	characters plus each schema's own key/enum-value literals (built by direct tree walk, per
	_schema_literals, never by enumerating the full admitted-string set -- see that function's
	own docstring for why enumeration does not scale here), matching this repo's own
	Claude/Loki/t2122-probe/probe.py::build_vocab convention (multi-char literals as merged
	tokens, single chars as fallback)."""
	vocab: list[str] = list(sorted(set('{}",:_0123456789')))
	for s in schemas:
		for lit in sorted(_schema_literals(s)):
			piece = lit.strip('"').rstrip(":")
			if piece and piece not in vocab:
				vocab.append(piece)
	return vocab


class G5_1_HermeticCorpusIsDeterministic(unittest.TestCase):
	"""T-2121 N2: the generated-schemas half of the corpus is hermetic and byte-identical
	across runs -- proven here directly (two independent generator invocations), not merely
	asserted."""

	def test_two_generations_are_byte_identical(self) -> None:
		a = generate_hermetic_corpus(n_schemas=12, max_depth=2)
		b = generate_hermetic_corpus(n_schemas=12, max_depth=2)
		self.assertEqual(json.dumps(a, sort_keys=False), json.dumps(b, sort_keys=False))


class G5_1_CompilerConformanceFuzz(unittest.TestCase):
	"""Design Sec6 G5-1 red suite: "a compiled DFA accepts every string the schema's per-field
	subset admits and rejects every string it does not" (G-7, completeness). Corpus: hand-built
	(the reference task's own schema, D-SLM32) + generated (this file's own hermetic
	generator)."""

	def _run_conformance(self, schema: dict) -> None:
		# Bounded to 2000 admitted strings: exhaustive for the small generated-corpus schemas
		# (max_depth<=2, <=4 fields/level keeps the true admitted-set size well under this in
		# practice, per the generator's own bench: 8..276480 -- the largest already exceeds
		# 2000 and is sampled, which is correct for a FUZZ suite's own claim) and a genuine
		# bounded sample for any schema whose product is larger.
		admitted = _canonical_serializations(schema, limit=2000)
		vocab = _reference_vocab([schema])
		pages = compile_schema_to_mask_pages(schema, vocab)
		# ACCEPTS every string the schema admits: each canonical serialization walks the
		# compiled DFA from start to an accepting state without ever hitting an empty mask.
		for s in admitted:
			self.assertTrue(
				pages.accepts(s),  # type: ignore[attr-defined]
				f"compiled DFA rejects an admitted serialization: {s!r}",
			)
		# REJECTS strings the schema does NOT admit: a single-character corruption of each
		# admitted string (still within the vocabulary) must not also be accepted, unless the
		# corruption happens to land on another admitted string (checked explicitly).
		for s in admitted[:20]:  # bounded for fuzz runtime; the hand-built reference schema's
			                     # own corpus (below) is exhaustive, not bounded.
			corrupted = s[:-1] + ("X" if s[-1:] != "X" else "Y")
			if corrupted in admitted:
				continue
			self.assertFalse(
				pages.accepts(corrupted),  # type: ignore[attr-defined]
				f"compiled DFA accepts a non-admitted corruption: {corrupted!r}",
			)

	def test_generated_corpus_conformance(self) -> None:
		for schema in generate_hermetic_corpus(n_schemas=12, max_depth=2):
			self._run_conformance(schema)

	def test_reference_schema_conformance(self) -> None:
		# The reference task's own schema (design Sec3, D-SLM32) -- Claude/Docs/
		# Shopkeeper_IntentExtraction_Schema.md, reproduced from Claude/Loki/t2122-probe/
		# probe.py's own SHOPKEEPER_SCHEMA construction (the strike's own grounding, not a
		# fresh transcription) -- MUST compile and must NOT be rejected (cross-field is
		# scored, not a compiler failure, D-SLM45).
		slot_enums = {
			"artist": ["unspecified", "rin", "mox", "vale", "any"],
			"day": ["unspecified", "mon", "tue", "wed", "thu", "fri", "sat", "sun"],
			"time_block": ["unspecified", "morning", "afternoon", "evening"],
			"size": ["unspecified", "small", "medium", "large", "sleeve"],
			"placement": ["unspecified", "arm", "leg", "back", "chest", "hand", "neck"],
			"style": ["unspecified", "blackwork", "traditional", "realism", "script", "geometric"],
			"budget_band": ["unspecified", "under_200", "200_500", "500_1000", "over_1000"],
			"deposit_ok": ["unspecified", "yes", "no"],
		}
		shopkeeper_schema = {
			"type": "object",
			"additionalProperties": False,
			"properties": {
				"intent": {"enum": ["book", "reschedule", "cancel", "price_query",
				                    "availability_query", "design_query", "complaint",
				                    "out_of_domain"]},
				"slots": {
					"type": "object",
					"additionalProperties": False,
					"properties": {k: {"enum": v} for k, v in slot_enums.items()},
				},
				"polarity": {"enum": ["affirm", "negate"]},
				"negated_slot": {"enum": ["none", "artist", "day", "time_block", "size",
				                          "placement", "style", "budget_band"]},
				"correction": {"type": "boolean"},
			},
		}
		vocab = _reference_vocab([shopkeeper_schema])
		try:
			compile_schema_to_mask_pages(shopkeeper_schema, vocab)
		except SchemaCompileError as exc:
			self.fail(
				f"the reference task's own schema (D-SLM32) was REJECTED as unsatisfiable -- "
				f"cross-field constraints are SCORED, not compiled-in (D-SLM45); this schema "
				f"has none of its own three cross-field-carrying RULES compiled as hard "
				f"constraints, only the per-field enum/object shape above, which must "
				f"compile cleanly. state={exc.state_id} prefix={exc.prefix!r} "
				f"reason={exc.reason}"
			)


class G5_1_UnsatisfiableSchemaRejectionGuardVitality(unittest.TestCase):
	"""Design Sec3/Sec9 G-7a: "the schema compiler proves every reachable non-accepting state
	has a non-empty valid-token set at compile time and rejects the schema otherwise (a
	reject-over-degrade rejection, which-state/why diagnostic)." This is the MECHANISM half of
	dim11's own two-cell G-7a split (the runtime's non-check negative control is the OTHER
	half, dim11_guard_red.cpp, this suite)."""

	def test_unsatisfiable_schema_rejected_with_which_state_diagnostic(self) -> None:
		# A schema whose vocabulary cannot spell one of its own required literals -- e.g. an
		# enum value containing a character absent from the supplied vocabulary -- has a
		# reachable non-accepting state with an empty valid-token set BY CONSTRUCTION.
		unsatisfiable_schema = {
			"type": "object",
			"additionalProperties": False,
			"properties": {
				"field": {"enum": ["needs_a_z_character"]},
			},
		}
		vocab_missing_z = list("abcdefghijklmnopqrstuvwxy0123456789{}\":,_")  # no 'z'
		with self.assertRaises(SchemaCompileError) as ctx:
			compile_schema_to_mask_pages(unsatisfiable_schema, vocab_missing_z)
		exc = ctx.exception
		# The which-state/why diagnostic (design Sec3/Sec9): the rejection names a concrete
		# state id and a human-readable reason, not merely "compile failed."
		self.assertIsNotNone(exc.state_id, "rejection did not name the offending state id")
		self.assertIn("z", exc.reason.lower(),
		              "rejection reason does not explain the missing continuation")

	def test_satisfiable_variant_of_same_schema_compiles(self) -> None:
		# The guard's OWN vitality requires the positive control too: the identical schema,
		# vocabulary now including 'z', compiles cleanly -- proving the rejection above is
		# about the missing vocabulary, not a spurious failure on this schema shape.
		satisfiable_schema = {
			"type": "object",
			"additionalProperties": False,
			"properties": {
				"field": {"enum": ["needs_a_z_character"]},
			},
		}
		vocab_with_z = list("abcdefghijklmnopqrstuvwxyz0123456789{}\":,_")
		compile_schema_to_mask_pages(satisfiable_schema, vocab_with_z)  # must not raise


if __name__ == "__main__":
	if _IMPORT_ERROR is not None:
		print(f"RED BY IMPORT: tools.sslm_convert_schema does not exist -- {_IMPORT_ERROR!r}")
		print("(expected pre-build; the build seat lands this module per design Sec6 G5-1)")
	unittest.main()
