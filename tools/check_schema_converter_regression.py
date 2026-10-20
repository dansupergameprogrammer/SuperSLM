"""check_schema_converter_regression.py -- byte-for-byte no-regression check for the schema compiler
(T-3094 S1, plan of record `Claude/Plans/superslm-multicommand-array/plan.md` Sec4.1, Wizard repo).

The rule this checks: every schema the baseline revision of `tools/sslm_convert_schema.py` (default
`origin/main`, which is v1.13.1's converter) compiles must still compile, to the same `MaskPages` table and
the same SCM1 bytes -- unless its table has a reachable state that cannot finish, which the current
converter refuses by name (the dead-state refusal outranks R1: generation from such a table could never
complete a document). The current converter meets it because its release limits bind only schemas the
compiler builds an array for (S1 fix round 1, R1); the at-limit corpus below holds array-free schemas at
or above every release limit, including ones that spell `{"type": "array"}` inside `enum`, `examples` or
`default`, so a limit that leaks onto them shows as DIFFERENT. This script compiles every case through
both converters and compares:

- the table: `start`, `accepting`, every state's row (`transitions`) and every state's packed mask page;
- the SCM1 section: the baseline's own writer (`tools/t2132_build_g5_fixture.py::_serialize_scm1` at the
  baseline revision) over the baseline table, against both that writer and the current
  `sslm_convert_schema.serialize_scm1` over the current table;
- a refusal: where the baseline refuses a case, the current converter must refuse it too;
- a dead-state refusal (`dead-state-refused`): the baseline compiles, the current converter refuses with
  its dead-end diagnostic, and the baseline's own table is checked here, independently, to have a
  reachable state from which no accepting state is reachable. Anything else is DIFFERENT.

Cases come from two sources:

1. The fixed corpus: the frozen G1 and G1g schemas, the T-2132 shopkeeper and one-field schemas, the
   T-2922/T-2909 string-leaf schema, and the S8/decode-threading one-token schema with its own
   vocabulary; each schema at the 256-byte vocabulary and, when present, at the pinned real tokenizer
   (`--tokenizer-dir`, special ids excluded).
   Plus the at-limit corpus, array-free schemas at or above each release limit that the baseline
   compiles: the code review's 200-value enum (23-byte values, 4,600 literal bytes), a 9-deep object
   chain, a 5,120-byte shared-prefix enum, a 4,200-byte distinct-prefix enum (over the DFA-state
   limit), compound (array and object) enum values, and the frozen array-free resource-pair over
   schemas (`tests/t3094-multicommand-s0/v2/resource-pairs-v2.json`: depth, NFA, DFA, transitions,
   section) at the pinned vocabulary.
2. Every `compile_schema_to_mask_pages` call an existing test suite makes, recorded by running that
   suite under pytest with the compiler wrapped under both import names it is loaded by
   (`sslm_convert_schema` and `tools.sslm_convert_schema`) (`--record`; default suites below). Each distinct
   (schema, vocabulary, special ids) call is replayed through both converters.

A schema containing an array that the baseline refuses is counted and skipped: the rule is about the
schemas the baseline compiles. One the baseline compiles (an `enum` of array values under
`"type": "array"`) is compared like any other.

    python tools/check_schema_converter_regression.py [--baseline-ref origin/main] [--record]
        [--suite PATH ...] [--tokenizer-dir DIR]

Exit status 0 when every compared case is identical; 1 otherwise.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DEFAULT_SUITES = (
    "tools/",
    "tests/reference/",
    "tests/t2130-g5-red-suite/g5_1_schema_compiler_fuzz_red.py",
    "tests/t2130-g5-red-suite/t2899_string_leaf_red.py",
    "tests/t2130-g5-red-suite/t2908_byte_level_red.py",
    "tests/t2130-g5-red-suite/t2910_boundary_red.py",
    "tests/t2130-g5-red-suite/t2912_value_level_red.py",
    "tests/t2130-g5-red-suite/t2916_c1_special_exclusion_red.py",
    "tests/t2130-g5-red-suite/t2916_s1_keyword_allowlist_red.py",
    "tests/t2130-g5-red-suite/t2916_s2_str_vocab_refused_red.py",
    "tests/t2130-g5-red-suite/t2919_te372_s2_nullable_enum_red.py",
    "tests/t2130-g5-red-suite/t2920_te372_s2_list_type_mismatch_pin.py",
    "tests/t2130-g5-red-suite/t2921_frozen_fixture_integrity.py",
    "tests/t2130-g5-red-suite/t2922_schema_query_contract.py",
)
DEFAULT_TOKENIZER = Path(os.environ.get("T3094_TOKENIZER_DIR", "/mnt/project-files/models/qwen2.5-0.5b-instruct-7ae5576"))
REAL_VOCAB_SIZE = 151936


# ------------------------------------------------------------------ the two converters

def _git_show(ref: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:{path}"], check=True,
                          capture_output=True, text=True).stdout


def _module_from_source(name: str, source: str, workdir: Path):
    path = workdir / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_converters(ref: str, workdir: Path):
    """(baseline converter, baseline SCM1 writer, current converter). The baseline modules are the
    files at `ref`, loaded under their own names; the current converter is the working tree's."""
    baseline = _module_from_source("baseline_sslm_convert_schema", _git_show(ref, "tools/sslm_convert_schema.py"), workdir)
    writer = _module_from_source("baseline_t2132_build_g5_fixture",
                                 _git_show(ref, "tools/t2132_build_g5_fixture.py"), workdir)
    import sslm_convert_schema as current
    return baseline, writer._serialize_scm1, current


# ------------------------------------------------------------------ the corpus

def has_array(node) -> bool:
    if isinstance(node, dict):
        kind = node.get("type")
        if kind == "array" or (isinstance(kind, list) and "array" in kind):
            return True
        return any(has_array(v) for v in node.values())
    if isinstance(node, list):
        return any(has_array(v) for v in node)
    return False


def _literal_assignment(path: Path, name: str):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{path}: no literal {name}")


def _pinned_vocab(tokenizer_dir: Path):
    tokenizer = tokenizer_dir / "tokenizer.json"
    if not tokenizer.is_file():
        return None
    from convert_tokenizer import TokenizerTables
    tables = TokenizerTables(str(tokenizer_dir))
    vocab = tuple(tables.id_to_bytes[i] if i < len(tables.id_to_bytes) else b"" for i in range(REAL_VOCAB_SIZE))
    return vocab, frozenset(i for _, i in tables.specials)


def fixed_corpus(tokenizer_dir: Path):
    """(label, schema, vocab, special_ids) for the fixed corpus."""
    v2 = ROOT / "tests" / "t3094-multicommand-s0" / "v2"
    import t2132_build_g5_fixture as G5
    schemas = {
        "g1": json.loads((v2 / "g1.schema.json").read_text(encoding="utf-8")),
        "g1g": json.loads((v2 / "g1g.schema.json").read_text(encoding="utf-8")),
        "t2132_shopkeeper": G5.SHOPKEEPER_SCHEMA,
        "t2132_minimal_one_field": G5.MINIMAL_ONE_FIELD_SCHEMA,
        "t2922_t2909_prompt_result": _literal_assignment(
            ROOT / "tests" / "t2130-g5-red-suite" / "make_t2922_dual_schema_fixture.py", "SCHEMA"),
    }
    byte_vocab = tuple(bytes([i]) for i in range(256))
    pinned = _pinned_vocab(tokenizer_dir)
    out = []
    for label, schema in schemas.items():
        out.append((f"{label}@bytes", schema, byte_vocab, frozenset()))
        if pinned is not None:
            out.append((f"{label}@pinned", schema, pinned[0], pinned[1]))
    for label, schema in at_limit_schemas():
        out.append((f"at_limit/{label}@bytes", schema, byte_vocab, frozenset()))
        if pinned is not None:
            out.append((f"at_limit/{label}@pinned", schema, pinned[0], pinned[1]))
    if pinned is not None:
        pairs = json.loads((v2 / "resource-pairs-v2.json").read_text(encoding="utf-8"))["pairs"]
        for pair in pairs:
            over = _in_required_order(pair.get("over_schema"))
            if over is not None and not has_array(over):
                out.append((f"at_limit/{pair['pair']}_over@pinned", over, pinned[0], pinned[1]))
    # Confirmation pass 1, C2: a string leaf whose closing quote no token can follow to an accepting
    # state ('}' is spelled, but the content loop has no exit the vocabulary completes) -- the baseline
    # compiles a table that can never finish; the current converter refuses it by name.
    out.append(("dead_state/c2_brace_key@own", _obj({"{": {"type": "string"}}),
                (b"{", b'"', b":", b"}", b" "), frozenset()))
    one_token = (b'{"ok":true}',) + tuple(f"<unused-{i}>".encode("ascii") for i in range(1, 64))
    out.append(("s8_decode_threading_one_token@own", G5.MINIMAL_ONE_FIELD_SCHEMA, one_token, frozenset()))
    return out, pinned is not None


def _in_required_order(node):
    """The frozen pair file stores properties key-sorted; the converter reads them in `required` order
    (the suite's `cv2_fixtures.in_required_order` does the same)."""
    if isinstance(node, list):
        return [_in_required_order(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {k: _in_required_order(v) for k, v in node.items()}
    if isinstance(out.get("properties"), dict) and isinstance(out.get("required"), list):
        props = out["properties"]
        out["properties"] = {k: props[k] for k in out["required"] if k in props}
        out["properties"].update({k: v for k, v in props.items() if k not in out["properties"]})
    return out


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def at_limit_schemas():
    """(label, schema): array-free schemas at or above a release limit that the baseline compiles."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    enum200 = [f"xxxxxxxxxxxxxxxxxxxx{i:03d}" for i in range(200)]
    chain = {"type": "boolean"}
    for _ in range(9):
        chain = _obj({"k": chain})
    return [
        ("enum_200x23", _obj({"a": {"enum": enum200}})),
        ("object_chain_9_deep", chain),
        ("enum_64x80_shared_prefix", _obj({"a": {"type": "string",
                                                 "enum": ["p" * 70 + f"{i:02d}" + "q" * 8 for i in range(64)]}})),
        ("enum_42x100_distinct", _obj({"a": {"type": "string", "enum": [
            "".join(letters[(i * 7 + j * 3) % 26] for j in range(98)) + f"{i:02d}" for i in range(42)]}})),
        ("c1_enum_200_examples_array", _obj({"a": {"enum": enum200, "examples": [{"type": "array"}]}})),
        ("c1_enum_200_default_array", _obj({"a": {"enum": enum200},
                                            "b": {"type": "boolean", "default": {"type": "array"}}})),
        ("c1_enum_200_enum_of_array_schema", _obj({"a": {"enum": enum200}, "b": {"enum": [{"type": "array"}]}})),
        ("enum_compound_array", {"type": "array", "enum": [[1, 2]]}),
        ("enum_compound_untyped", {"enum": [[1, 2], {"k": 1}]}),
        ("enum_compound_object", {"type": "object", "enum": [{"k": 1}]}),
        ("enum_compound_in_property", _obj({"a": {"enum": [[1], {"b": None}]}})),
    ]


def record_suite_calls(suites, record_path: Path) -> dict:
    """Runs pytest over `suites` in a child process with `compile_schema_to_mask_pages` wrapped, and
    returns {calls, vocabs} of every distinct call it made."""
    plugin_dir = Path(tempfile.mkdtemp(prefix="schema-regression-plugin-"))
    (plugin_dir / "schema_call_recorder.py").write_text(_RECORDER, encoding="utf-8")
    env = dict(os.environ, SCHEMA_CALL_RECORD=str(record_path),
               PYTHONPATH=os.pathsep.join([str(plugin_dir), str(HERE), str(ROOT), os.environ.get("PYTHONPATH", "")]),
               PYTHONDONTWRITEBYTECODE="1")
    began = time.time()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-p", "schema_call_recorder", "-p", "no:cacheprovider",
                           "-q", *suites], cwd=str(ROOT), env=env,
                          capture_output=True, text=True)
    summary = (proc.stdout.strip().splitlines() or [""])[-1]
    print(f"recorded suites in {time.time() - began:.0f} s (pytest exit {proc.returncode}): {summary}")
    with open(record_path, "rb") as handle:
        recorded = pickle.load(handle)
    print(f"recorded {len(recorded['calls'])} distinct compile calls over {len(recorded['vocabs'])} vocabularies")
    return recorded


_RECORDER = r'''
"""pytest plugin: wraps sslm_convert_schema.compile_schema_to_mask_pages and records every distinct call."""
import copy, hashlib, os, pickle

_CALLS, _VOCABS = {}, {}


def _vocab_key(vocab):
    h = hashlib.sha256()
    for piece in vocab:
        h.update(len(piece).to_bytes(4, "little"))
        h.update(piece)
    return h.hexdigest()


def pytest_configure(config):
    import importlib
    for module_name in ("sslm_convert_schema", "tools.sslm_convert_schema"):
        _wrap(importlib.import_module(module_name))


def _wrap(SC):
    original = SC.compile_schema_to_mask_pages

    def recording(schema, vocab, *args, **kwargs):
        try:
            vocab_t = tuple(vocab)
            if all(isinstance(p, bytes) for p in vocab_t):
                vkey = _vocab_key(vocab_t)
                _VOCABS.setdefault(vkey, vocab_t)
                special = frozenset(kwargs.get("special_ids", frozenset()))
                snapshot = copy.deepcopy(schema)
                key = (pickle.dumps(snapshot), vkey, tuple(sorted(special)))
                _CALLS.setdefault(key, (snapshot, vkey, special))
        except Exception:
            pass
        return original(schema, vocab, *args, **kwargs)

    SC.compile_schema_to_mask_pages = recording


def pytest_unconfigure(config):
    with open(os.environ["SCHEMA_CALL_RECORD"], "wb") as handle:
        pickle.dump({"calls": list(_CALLS.values()), "vocabs": _VOCABS}, handle)
'''


# ------------------------------------------------------------------ the comparison

def _outcome(module, schema, vocab, special):
    try:
        return module.compile_schema_to_mask_pages(schema, vocab, special_ids=special), None
    except module.SchemaCompileError as error:
        return None, f"SchemaCompileError: {error}"
    except (TypeError, ValueError, KeyError, AttributeError, RecursionError) as error:
        return None, f"{type(error).__name__}: {error}"


EXPECTED = {"dead_state/c2_brace_key@own": "dead-state-refused"}
DEAD_END = "cannot reach an accepting state (dead end)"


def _cannot_finish(mp) -> list[int]:
    """Reachable states of a baseline table from which no accepting state is reachable."""
    reach, frontier = {mp.start}, [mp.start]
    while frontier:
        for target in mp.transitions.get(frontier.pop(), {}).values():
            if target not in reach:
                reach.add(target)
                frontier.append(target)
    back = {}
    for state, row in mp.transitions.items():
        for target in row.values():
            back.setdefault(target, set()).add(state)
    live, frontier = set(mp.accepting), list(mp.accepting)
    while frontier:
        for source in back.get(frontier.pop(), ()):
            if source not in live:
                live.add(source)
                frontier.append(source)
    return sorted(reach - live)


def compare(label, schema, vocab, special, baseline, baseline_writer, current) -> tuple[str, str]:
    """('identical' | 'both-refuse' | 'dead-state-refused' | 'DIFFERENT', detail)."""
    old, old_error = _outcome(baseline, schema, vocab, special)
    new, new_error = _outcome(current, schema, vocab, special)
    if old is None:
        if new is None:
            return "both-refuse", f"baseline {old_error[:120]!r} / current {new_error[:120]!r}"
        return "DIFFERENT", f"baseline refused ({old_error[:200]}) but the current converter compiles"
    if new is None:
        stuck = _cannot_finish(old)
        if DEAD_END in new_error and stuck:
            return "dead-state-refused", (f"baseline table has {len(stuck)} reachable state(s) that cannot finish "
                                          f"(e.g. {stuck[0]}); current refuses: {new_error[:160]}")
        return "DIFFERENT", f"baseline compiles but the current converter refuses: {new_error[:300]}"
    if (old.start, old.accepting) != (new.start, new.accepting):
        return "DIFFERENT", "start or accepting set differs"
    if old.transitions != new.transitions:
        return "DIFFERENT", "transition rows differ"
    states = sorted(set(old.transitions) | {old.start})
    if any(old.page(s) != new.page(s) for s in states):
        return "DIFFERENT", "a mask page differs"
    size = len(vocab)
    reference = baseline_writer([("schema", old)], size)
    if baseline_writer([("schema", new)], size) != reference:
        return "DIFFERENT", "SCM1 (baseline writer) differs"
    if current.serialize_scm1([("schema", new)], size) != reference:
        return "DIFFERENT", "SCM1 (current serialize_scm1) differs from the baseline writer's bytes"
    digest = hashlib.sha256(reference).hexdigest()[:16]
    return "identical", f"{len(states)} states, {sum(map(len, new.transitions.values()))} transitions, SCM1 {len(reference)} B sha256 {digest}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--baseline-ref", default="origin/main")
    ap.add_argument("--record", action="store_true", help="also replay every compile call the --suite runs make")
    ap.add_argument("--suite", action="append", default=None, help=f"pytest paths to record (default: {' '.join(DEFAULT_SUITES)})")
    ap.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    workdir = Path(tempfile.mkdtemp(prefix="schema-regression-"))
    baseline, baseline_writer, current = load_converters(args.baseline_ref, workdir)
    corpus, pinned = fixed_corpus(args.tokenizer_dir)
    print(f"baseline {args.baseline_ref}: {baseline.__file__}")
    print(f"current: {current.__file__}")
    if not pinned:
        print(f"pinned tokenizer absent at {args.tokenizer_dir}: fixed corpus runs at the byte vocabulary only")

    if args.record:
        recorded = record_suite_calls(args.suite or list(DEFAULT_SUITES), workdir / "calls.pickle")
        for index, (schema, vkey, special) in enumerate(recorded["calls"]):
            corpus.append((f"recorded#{index}", schema, recorded["vocabs"][vkey], special))

    tally = {"identical": 0, "both-refuse": 0, "dead-state-refused": 0, "DIFFERENT": 0, "skipped-array": 0}
    for label, schema, vocab, special in corpus:
        if has_array(schema) and _outcome(baseline, schema, vocab, special)[0] is None:
            tally["skipped-array"] += 1  # the baseline refuses it: an array schema, outside the rule
            continue
        verdict, detail = compare(label, schema, vocab, special, baseline, baseline_writer, current)
        if label in EXPECTED and verdict != EXPECTED[label]:
            verdict, detail = "DIFFERENT", f"expected {EXPECTED[label]}, got {verdict}: {detail}"
        tally[verdict] += 1
        if verdict == "DIFFERENT" or args.verbose or not label.startswith("recorded#"):
            print(f"{verdict:12} {label}: {detail}")
    print(json.dumps(tally, sort_keys=True))
    return 1 if tally["DIFFERENT"] else 0


if __name__ == "__main__":
    sys.exit(main())
