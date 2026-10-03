"""T-3094 S0 compiler red suite. Run `python -m pytest ... -q`.

Reference-control tests are green by design. Product cells must fail their own
assertion against the 1.12 converter. An absent future API is explicitly
red-unimplemented and never credited as a killed mutant.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))
from sslm_convert_schema import SchemaCompileError, compile_schema_to_mask_pages  # noqa: E402
from reference import (canonical_walk, consumer_contract, grade_pc2, parse_scm1,
                       parse_shape, scm1_account, write_scm1, grade_semantics, semantic_pass)  # noqa: E402
from resource_pairs import DIAGNOSTICS, pair  # noqa: E402
from keyword_catalogue import ALLOWED, ANNOTATIONS, KEYWORDS, VALUES, node  # noqa: E402
from diversity import BANNED_TRACE_FIELDS, census, gate, rows as corpus_rows, trace_keys  # noqa: E402
from assemble_corpus import build as authored_population  # noqa: E402
from raw_identifiability import audit as raw_audit, reading as raw_reading  # noqa: E402

SCHEMA = json.loads((HERE / "command.schema.json").read_text(encoding="utf-8"))
BYTE_VOCAB = tuple(bytes([i]) for i in range(256))
FIELD = ("action", "subject", "target", "trigger", "depends_on_previous")


def command(action="fire", subject="alpha", target="ridge", trigger="signal", dependency=False):
    return {"action": action, "subject": subject, "target": target,
            "trigger": trigger, "depends_on_previous": dependency}


def serialized(items):
    return ('{"commands":[' + ",".join(json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                                    for item in items) + "]}").encode("utf-8")


def byte_pieces(raw):
    return tuple(bytes([byte]) for byte in raw)


def example(node):
    if "enum" in node:
        return node["enum"][0]
    if node["type"] == "boolean":
        return False
    if node["type"] == "object":
        return {key: example(child) for key, child in node["properties"].items()}
    if node["type"] == "array":
        return [example(node["items"]) for _ in range(node.get("minItems", 0))]
    raise ValueError(f"no finite example: {node}")


def walk(mp, pieces):
    state = mp.start
    for piece in pieces:
        token_id = BYTE_VOCAB.index(piece)
        assert token_id in mp.valid_token_ids(state), f"missing token {piece!r} at {state}"
        state = mp.step(state, token_id)
    assert mp.is_accepting(state), "completed document did not reach accepting state"


def test_frozen_population_and_hashes():
    manifest = json.loads((HERE / "corpus-manifest.json").read_text())
    rows = [json.loads(line) for line in (HERE / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    assert hashlib.sha256((HERE / "corpus.jsonl").read_bytes()).hexdigest() == manifest["sha256"]
    for key, file in (("guide_sha256", "annotation-guide.md"),
                      ("baseline_prompt_sha256", "baseline-prompt.txt"),
                      ("schema_sha256", "command.schema.json"),
                      ("freeze_sha256", "freeze.json"),
                      ("resource_counts_sha256", "resource-model-counts.json"),
                      ("annotation_input_sha256", "annotation-input.jsonl"),
                      ("blind_annotation_input_sha256", "blind-annotation-input.jsonl"),
                      ("model_input_sha256", "model-input.jsonl"),
                      ("identifiability_report_sha256", "identifiability-report.json"),
                      ("diversity_report_sha256", "diversity-report.json"),
                      ("assembler_sha256", "assemble_corpus.py"),
                      ("diversity_script_sha256", "diversity.py"),
                      ("identifiability_script_sha256", "raw_identifiability.py"),
                      ("attributes_sha256", ".gitattributes"),
                      ("keyword_catalogue_sha256", "keyword_catalogue.py")):
        assert hashlib.sha256((HERE / file).read_bytes()).hexdigest() == manifest[key]
    assert all(hashlib.sha256((HERE / file).read_bytes()).hexdigest() == digest
               for file, digest in manifest["authored_source_sha256"].items())
    assert json.loads((HERE / "freeze.json").read_text())["version"] == "T-3094-S0-v4"
    assert len(rows) == manifest["rows"] == 1960
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["utterance"] for r in rows}) == len(rows)
    assert sum(r["split"] == "held_out" for r in rows) == 1760
    assert sum(r["split"] == "development" for r in rows) == 200
    assert all(r["provenance"] == "individually_authored_synthetic" for r in rows)
    assert not ({r["source_session"] for r in rows if r["split"] == "held_out"} &
                {r["source_session"] for r in rows if r["split"] == "development"})
    assert not ({r["speaker_id"] for r in rows if r["split"] == "held_out"} &
                {r["speaker_id"] for r in rows if r["split"] == "development"})
    annotation = [json.loads(line) for line in (HERE / "annotation-input.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(annotation) == 1760
    assert {a["case_id"] for a in annotation} == {r["annotation_case_id"] for r in rows if r["split"] == "held_out"}
    assert all(set(a) == {"case_id", "utterance", "game_action_trace"} for a in annotation)
    blind = [json.loads(line) for line in (HERE / "blind-annotation-input.jsonl").read_text(encoding="utf-8").splitlines()]
    model = [json.loads(line) for line in (HERE / "model-input.jsonl").read_text(encoding="utf-8").splitlines()]
    held = [r for r in rows if r["split"] == "held_out"]
    assert all(set(b) == {"case_id", "utterance"} for b in blind)
    assert all(set(m) == {"utterance"} for m in model)
    assert [b["case_id"] for b in blind] == [r["annotation_case_id"] for r in held]
    assert [m["utterance"] for m in model] == [r["utterance"] for r in held]
    assert not any(key in BANNED_TRACE_FIELDS for r in rows for key in trace_keys(r["game_action_trace"]))
    counts = manifest["counts"]
    assert all(counts[name] == 40 for name in ("at_cap", "cap_plus_one", "ambiguous_executable"))
    assert counts["pc2"] == 240
    assert counts["cancellation"] == counts["zero_action"] == 200  # 100 general + 100 dedicated
    pairs = [r["pair"] for r in rows if r["pair"]]
    assert len(pairs) == 240 and all(n == 2 for n in __import__("collections").Counter(pairs).values())


def test_v4_diversity_freeze_and_report():
    frozen = json.loads((HERE / "freeze.json").read_text())["diversity_gates"]
    report = census(corpus_rows())
    assert report == {k: v for k, v in json.loads((HERE / "diversity-report.json").read_text()).items()
                      if k != "gate_failures"}
    assert gate(report, frozen) == []


def test_raw_only_wait_identifiability_and_v3_negative_controls():
    rows = corpus_rows()
    report = raw_audit(rows)
    assert report == json.loads((HERE / "identifiability-report.json").read_text())
    assert report["rows"] == 320 and report["passed"] and report["failure_count"] == 0
    assert raw_audit([{k: v for k, v in row.items() if k != "game_action_trace"} for row in rows]) == report
    for old_raw in (
        "Alpha, stay on the clock for green; fire into north ridge when it comes.",
        "Alpha, stay on the clock for green, then fire into north ridge.",
        "Alpha wing, north ridge, scan, hold, keep the green watch, then fire.",
    ):
        assert raw_reading(old_raw) == "ambiguous"


def test_authored_population_repackages_exactly():
    assert authored_population() == corpus_rows()


@pytest.mark.parametrize("guard", tuple(DIAGNOSTICS))
def test_resource_model_counts_frozen(guard):
    _under, _over, below, above = pair(guard)
    frozen = json.loads((HERE / "resource-model-counts.json").read_text())
    assert frozen[guard] == {"under": below, "over": above}
    assert below[guard] < above[guard]


def test_unique_subcases_discriminate_their_named_action_semantics():
    rows = [json.loads(line) for line in (HERE / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    unique = {name: [r for r in rows if r["id"].startswith(f"U-{name}-")]
              for name in ("correction", "shared_argument", "pronoun", "nested_condition",
                           "parallel", "at_cap", "cap_plus_one")}
    assert all(len(part) == 40 for part in unique.values())
    def events(row, name):
        return [e for e in row["game_action_trace"]["events"] if e["event"] == name]
    assert all(events(r, "utterance_revision") and
               events(r, "utterance_revision")[0]["discarded_place"] != r["game_action_trace"]["target"]
               for r in unique["correction"])
    assert all([e["task"] for e in events(r, "task_queued")] == ["scan", "hold"]
               for r in unique["shared_argument"])
    assert all([e["task"] for e in events(r, "task_queued")] == ["move", "scan"]
               for r in unique["pronoun"])
    nested = unique["nested_condition"]
    assert all({e["name"] for e in events(r, "condition_source_registered")} ==
               {"area_clear", "scout_confirmation"} for r in nested)
    assert all(len({r["game_action_trace"]["actor"],
                    events(r, "task_queued")[1]["actor"]}) == 2
               for r in unique["parallel"])
    assert all([e["task"] for e in events(r, "task_queued")] ==
               ["move", "scan", "hold", "fire"] and events(r, "signal_listener_registered")
               for r in unique["at_cap"])
    assert all([e["task"] for e in events(r, "task_queued")] ==
               ["move", "scan", "hold", "signal_watch", "fire"] and
               events(r, "completion_callback_registered") for r in unique["cap_plus_one"])
    for condition in ("simple_trigger", "dependency", "ordering"):
        pairs = {}
        for r in rows:
            if r["pair"] and r["pair"].startswith("P-" + condition + "-"):
                pairs.setdefault(r["pair"], []).append(r)
        assert len(pairs) == 40
        for members in pairs.values():
            assert len(members) == 2
            a, b = members
            assert (a["game_action_trace"]["actor"], a["game_action_trace"]["target"]) == (
                b["game_action_trace"]["actor"], b["game_action_trace"]["target"])
            assert {r["id"][-1] for r in members} == {"T", "W"}
            assert sum(bool(events(r, "completion_callback_registered")) for r in members) == 1
            assert sum(bool(events(r, "signal_listener_registered")) for r in members) == 1


@pytest.mark.parametrize("raw,shape_ok,canonical_ok", [
    (serialized([command()]), True, True),
    (serialized([command(subject="café", target="ridge\\gate")]), True, True),
    (b'{"commands":[{"subject":"alpha","action":"fire","target":"ridge","trigger":"signal","depends_on_previous":false}]}', True, False),
    (b'{ "commands":[{"action":"fire","subject":"alpha","target":"ridge","trigger":"signal","depends_on_previous":false}]}', True, False),
    (serialized([command(subject=" ")]), True, False),
    (serialized([command()])[:-2] + b',]}', False, False),
    (serialized([command()])[:-2], False, False),
])
def test_oracle_a_b_independent_controls(raw, shape_ok, canonical_ok):
    def accepted(fn):
        try:
            fn()
            return True
        except (ValueError, UnicodeError, json.JSONDecodeError):
            return False
    assert accepted(lambda: parse_shape(raw)) is shape_ok
    assert accepted(lambda: canonical_walk(raw, byte_pieces(raw))) is canonical_ok


def test_compound_token_directions_reference():
    raw = serialized([command()])
    opening = raw.index(b'"alpha"')
    # Structure-to-content: token includes the opening quote and first content byte.
    bad = byte_pieces(raw[:opening]) + (b'"a',) + byte_pieces(raw[opening + 2:])
    with pytest.raises(ValueError, match="compound opening"):
        canonical_walk(raw, bad)
    # Content-to-structure: last content byte and closing quote may share a token.
    closing = raw.index(b'alpha"') + len(b'alpha') - 1
    good = byte_pieces(raw[:closing]) + (b'a"',) + byte_pieces(raw[closing + 2:])
    assert canonical_walk(raw, good) == parse_shape(raw)


def test_independent_keyword_catalogue_is_closed():
    assert KEYWORDS == set(VALUES)
    assert set(ALLOWED) == {"root", "array", "item", "enum", "subject", "target", "trigger", "boolean"}
    assert all(allowed <= KEYWORDS for allowed in ALLOWED.values())
    assert all(ANNOTATIONS <= allowed for allowed in ALLOWED.values())
    assert sum(len(KEYWORDS - allowed) for allowed in ALLOWED.values()) >= 400


def test_scm1_reference_corruption():
    # One state, one token self-loop; actual wire format and independent offsets.
    data = (struct.pack("<4sIIIII", b"SCM1", 1, 1, 2, 1, 0)
            + struct.pack("<IIIIIIQQQQ", 0, 1, 1, 1, 1, 0, 81, 82, 90, 94)
            + b"x" + b"\x01" + struct.pack("<II", 0, 1) + struct.pack("<I", 0)
            + struct.pack("<II", 0, 0))
    assert len(data) == scm1_account(1, 1, 1, 2)["total"]
    assert parse_scm1(data)["transitions"] == 1
    corrupt = bytearray(data)
    corrupt[81] = 0
    with pytest.raises(ValueError, match="mask"):
        parse_scm1(bytes(corrupt))
    assert write_scm1("x", 2, {0: {0: 0}}, frozenset({0})) == data


def test_consumer_reference_atomicity():
    valid = serialized([command(), command(target="bridge")])
    assert consumer_contract(valid, route="Executable", stop="Complete", accepting=True, cap=1) == ("OverCap", (), 1, 0)
    assert consumer_contract(valid, route="Executable", stop="Budget", accepting=False, cap=2) == ("BudgetStop", (), 1, 0)
    assert consumer_contract(valid, route="Executable", stop="DeadEnd", accepting=False, cap=2) == ("SchemaDeadEnd", (), 1, 0)
    assert consumer_contract(valid, route="Cancelled", stop="Complete", accepting=True, cap=2) == ("Cancelled", (), 0, 0)
    assert consumer_contract(valid, route="NoAction", stop="Complete", accepting=True, cap=2) == ("NoAction", (), 0, 0)
    assert consumer_contract(valid, route="Executable", stop="Complete", accepting=True, cap=2)[0] == "Accepted"


def test_pc2_reference_discrimination():
    rows = corpus_rows()
    for condition in ("simple_trigger", "dependency", "ordering"):
        members = {}
        for row in rows:
            if row["pair"] and row["pair"].startswith("P-" + condition + "-"):
                members.setdefault(row["pair"], {})[row["id"][-1]] = row
        assert len(members) == 40
        gold = []
        for arms in members.values():
            def event_commands(row):
                trace = row["game_action_trace"]
                tasks = [event["task"] for event in trace["events"] if event["event"] == "task_queued"]
                return tuple(command("wait" if task == "signal_watch" else task,
                                     subject=trace["actor"], target=trace["target"],
                                     dependency=task == "fire" and "signal_watch" in tasks)
                             for task in tasks)
            gold.append((event_commands(arms["T"]), event_commands(arms["W"])))
        assert len(set(str(pair) for pair in gold)) == 40  # actor/target matched but varied across pairs
        assert grade_pc2(gold, gold)["passes"]
        assert not grade_pc2(gold, [(right, right) for _, right in gold])["passes"]
        assert not grade_pc2(gold, [(left, left) for left, _ in gold])["passes"]


def test_semantic_reference_valid_json_error_and_no_action_hallucination():
    gold = ([{"id": f"e{i}", "status": "Executable", "commands": [command()]} for i in range(100)]
            + [{"id": f"cancel{i}", "status": "Cancelled", "commands": []} for i in range(100)]
            + [{"id": f"zero{i}", "status": "NoAction", "commands": []} for i in range(100)])
    predicted = [{**r, "query_starts": 1 if r["status"] == "Executable" else 0,
                  "action_attempts": 0} for r in gold]
    assert semantic_pass(grade_semantics(gold, predicted))
    count_shift = [dict(r) for r in predicted]
    for r in count_shift[:6]:
        r["commands"] = [command(), command()]
    assert not semantic_pass(grade_semantics(gold, count_shift))
    hallucination = [dict(r) for r in predicted]
    hallucination[-1].update(commands=[command()])
    assert not semantic_pass(grade_semantics(gold, hallucination))
    wrong_status = [dict(r) for r in predicted]
    for r in wrong_status[:6]:
        r["status"] = "NoAction"
    assert not semantic_pass(grade_semantics(gold, wrong_status))


def test_existing_converter_rejects_array_for_array_construct():
    with pytest.raises(SchemaCompileError, match="array|unsupported construct"):
        compile_schema_to_mask_pages(SCHEMA, BYTE_VOCAB, special_ids=frozenset())


@pytest.mark.parametrize("minimum", [0, 1, 2])
def test_cyclic_array_accepts_exact_minimum_and_repeats(minimum):
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["commands"]["minItems"] = minimum
    try:
        mp = compile_schema_to_mask_pages(schema, BYTE_VOCAB, special_ids=frozenset())
    except SchemaCompileError as error:
        pytest.fail(f"red-unimplemented array compiler: {error}")
    for count in range(minimum, minimum + 3):
        walk(mp, byte_pieces(serialized([command(target=f"ridge{i}") for i in range(count)])))


def test_array_language_against_two_independent_oracles_and_wire():
    valid = serialized([command(subject="café"), command(subject="alpha\\bravo")])
    assert canonical_walk(valid, byte_pieces(valid)) == parse_shape(valid)
    try:
        mp = compile_schema_to_mask_pages(SCHEMA, BYTE_VOCAB, special_ids=frozenset())
    except SchemaCompileError as error:
        pytest.fail(f"red-unimplemented array language and SCM1: {error}")
    walk(mp, byte_pieces(valid))
    encoded = write_scm1("commands", len(BYTE_VOCAB), mp.transitions, mp.accepting)
    parsed = parse_scm1(encoded)
    assert parsed["states"] == len(mp.transitions)
    assert parsed["transitions"] == sum(map(len, mp.transitions.values()))
    invalid = (
        b'{"commands":[]}',
        b'{"commands":[{"subject":"alpha","action":"fire","target":"ridge","trigger":"signal","depends_on_previous":false}]}',
        b'{ "commands":[' + valid.split(b'[', 1)[1],
        valid[:-2] + b',]}',
        valid[:-1],
        valid[:-2] + b',"extra":1}]}',
    )
    for raw in invalid:
        with pytest.raises((ValueError, AssertionError)):
            canonical_walk(raw, byte_pieces(raw))
        with pytest.raises(AssertionError):
            walk(mp, byte_pieces(raw))


def test_compound_opening_masked_and_closing_enabled_on_actual_rows():
    vocab = BYTE_VOCAB + (b'"a', b'a"')
    try:
        mp = compile_schema_to_mask_pages(SCHEMA, vocab, special_ids=frozenset())
    except SchemaCompileError as error:
        pytest.fail(f"red-unimplemented compound boundary array: {error}")
    raw = serialized([command(subject="alpha")])
    opening = raw.index(b'"alpha"')
    state = mp.start
    for byte in raw[:opening]:
        assert byte in mp.valid_token_ids(state)
        state = mp.step(state, byte)
    assert len(BYTE_VOCAB) not in mp.valid_token_ids(state), "compound opening must be masked"
    assert ord('"') in mp.valid_token_ids(state), "ordinary opening control"
    for byte in b'"alph':
        state = mp.step(state, byte)
    assert len(BYTE_VOCAB) + 1 in mp.valid_token_ids(state), "compound close must be enabled"
    closing_target = mp.step(state, len(BYTE_VOCAB) + 1)
    ordinary_target = mp.step(mp.step(state, ord('a')), ord('"'))
    assert closing_target == ordinary_target, "compound close destination"


@pytest.mark.parametrize("location,keyword", tuple(
    (location, keyword) for location, allowed in ALLOWED.items()
    for keyword in sorted(KEYWORDS - allowed)))
def test_keyword_is_refused_at_each_recursive_node(location, keyword):
    if location != "root":
        try:
            compile_schema_to_mask_pages(SCHEMA, BYTE_VOCAB, special_ids=frozenset())
        except SchemaCompileError as error:
            pytest.fail(f"red-unimplemented array baseline before {location}/{keyword} refusal: {error}")
    schema = json.loads(json.dumps(SCHEMA))
    node(schema, location)[keyword] = VALUES[keyword]
    with pytest.raises(SchemaCompileError, match=re_escape(keyword)):
        compile_schema_to_mask_pages(schema, BYTE_VOCAB, special_ids=frozenset())


@pytest.mark.parametrize("location", tuple(ALLOWED))
@pytest.mark.parametrize("annotation", tuple(sorted(ANNOTATIONS)))
def test_annotations_accepted_at_every_supported_node(location, annotation):
    schema = json.loads(json.dumps(SCHEMA))
    annotation_value = {"examples": ["sample"], "deprecated": False,
                        "readOnly": False, "writeOnly": False}.get(annotation, "metadata")
    node(schema, location)[annotation] = annotation_value
    try:
        mp = compile_schema_to_mask_pages(schema, BYTE_VOCAB, special_ids=frozenset())
    except SchemaCompileError as error:
        pytest.fail(f"red-unimplemented annotation at {location}: {error}")
    walk(mp, byte_pieces(serialized([command()])))


@pytest.mark.parametrize("keyword", ("$schema", "$id"))
def test_root_metadata_is_permitted(keyword):
    schema = json.loads(json.dumps(SCHEMA))
    schema[keyword] = VALUES[keyword]
    try:
        mp = compile_schema_to_mask_pages(schema, BYTE_VOCAB, special_ids=frozenset())
    except SchemaCompileError as error:
        pytest.fail(f"red-unimplemented root {keyword} acceptance: {error}")
    walk(mp, byte_pieces(serialized([command()])))


@pytest.mark.parametrize("defect,expected", (
    ("items_missing", "items"), ("items_tuple", "items"),
    ("min_negative", "minItems"), ("min_fraction", "minItems"),
    ("optional_item_key", "required"), ("reordered_item_key", "required"),
    ("extra_properties", "additionalProperties"),
))
def test_malformed_array_schema_named_refusal(defect, expected):
    try:
        compile_schema_to_mask_pages(SCHEMA, BYTE_VOCAB, special_ids=frozenset())
    except SchemaCompileError as error:
        pytest.fail(f"red-unimplemented array baseline before {defect} refusal: {error}")
    schema = json.loads(json.dumps(SCHEMA))
    array = schema["properties"]["commands"]
    item = array["items"]
    if defect == "items_missing":
        del array["items"]
    elif defect == "items_tuple":
        array["items"] = [item]
    elif defect == "min_negative":
        array["minItems"] = -1
    elif defect == "min_fraction":
        array["minItems"] = 1.5
    elif defect == "optional_item_key":
        item["required"].remove("target")
    elif defect == "reordered_item_key":
        item["required"] = list(reversed(item["required"]))
    else:
        item["additionalProperties"] = True
    with pytest.raises(SchemaCompileError, match=expected):
        compile_schema_to_mask_pages(schema, BYTE_VOCAB, special_ids=frozenset())


def re_escape(value):
    return __import__("re").escape(value)


@pytest.mark.parametrize("guard", tuple(DIAGNOSTICS))
def test_resource_pair_refuses_only_named_guard_without_output(guard):
    under, over, under_model, over_model = pair(guard)
    assert under_model[guard] < over_model[guard]
    frozen_counts = json.loads((HERE / "resource-model-counts.json").read_text())
    assert frozen_counts[guard] == {"under": under_model, "over": over_model}
    # The independent model and the injected limit are fixed before the
    # converter has an admission-counter API. S1 reconciles exact counts.
    limits = {name: 2**63 - 1 for name in DIAGNOSTICS}
    limits[guard] = under_model[guard]
    try:
        mp = compile_schema_to_mask_pages(under, BYTE_VOCAB, special_ids=frozenset(), limits=limits)
    except TypeError as error:
        pytest.fail(f"red-unimplemented {guard} admission API: {error}")
    raw = json.dumps(example(under), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    walk(mp, byte_pieces(raw))
    with pytest.raises(SchemaCompileError, match=DIAGNOSTICS[guard]):
        compile_schema_to_mask_pages(over, BYTE_VOCAB, special_ids=frozenset(), limits=limits)
