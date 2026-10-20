"""Item 5 — the offline schema attachment command (plan §4.1 attachment paragraph; §6 rows 2, 8, 9; §8 S1, S2).

The command does not exist yet, so every product cell here is red-unimplemented and is never credited as a
killed mutant. Contract assumed (the T-3094 attachment contract re-pointed to IntentDoc v2):
  python tools/sslm_attach_schema.py --base B --ckpt C --schema NAME:PATH [--schema NAME:PATH ...] --out O
                                     [--limits-json J]
Success is exit 0 with O written; a refusal is a non-zero exit, a diagnostic on stderr naming its reason,
and no file at O (no partial artifact)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import cv2_artifact as R
import cv2_cases as C
import cv2_oracle as B
import cv2_tokens as T
import cv2_attach_fixtures as AF
import cv2_fixtures as F
import cv2_injection as I
import count_model_v2 as cm

import os  # noqa: E402

# The command under test; CV2_ATTACH_TOOL points the cells at a mutant copy (mutant runs only).
TOOL = Path(os.environ.get("CV2_ATTACH_TOOL", F.TOOLS / "sslm_attach_schema.py"))
RED = "red-unimplemented: the supported offline schema attachment command tools/sslm_attach_schema.py is absent"


_BUILT = {}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    def get():
        if not (F.TOKENIZER_DIR / "tokenizer.json").is_file():
            pytest.skip(f"pinned tokenizer absent at {F.TOKENIZER_DIR} (set T3094_TOKENIZER_DIR)")
        if "fx" not in _BUILT:
            _BUILT["fx"] = AF.build(tmp_path_factory.mktemp("attach"))
        return _BUILT["fx"]
    return get


@pytest.fixture
def fx(built, tmp_path_factory):
    """Product cells: the command must exist before any fixture is built (red-unimplemented otherwise), and
    the healthy attach of IntentDoc v2 to the base must succeed first, so that no refusal cell is satisfied
    by a base the command (or its verifier) rejects on its own."""
    def get():
        _assert_tool()
        fixtures = built()
        if "healthy" not in _BUILT:
            out = tmp_path_factory.mktemp("healthy") / "healthy.sslm"
            result = run(fixtures["base"], fixtures["ckpt"], [("intents", fixtures["intentdoc"])], out)
            _BUILT["healthy"] = (result.returncode, out.exists(), result.stderr.decode("utf-8", "replace")[-800:])
        code, written, stderr = _BUILT["healthy"]
        assert code == 0 and written, f"healthy control: attaching IntentDoc v2 to the base failed: {stderr}"
        return fixtures
    return get


def _assert_tool():
    assert TOOL.is_file(), RED


def test_reference_fixtures_have_their_stated_properties(built):
    """Reference control: the fixtures are what the cells say they are, checked by the independent readers."""
    fx = built()
    from convert_tokenizer import TokenizerTables
    base = R.read_container(fx["base"].read_bytes())
    tok = [s["data"] for s in base["sections"] if s["type"] == R.TOKENIZER]
    assert len(tok) == 1 and tok[0] == bytes(TokenizerTables(str(fx["ckpt"])).serialize_tokenizer())
    wrong = bytes(TokenizerTables(str(fx["wrong_ckpt"])).serialize_tokenizer())
    assert wrong != tok[0] and len(TokenizerTables(str(fx["wrong_ckpt"])).id_to_bytes) == \
        len(TokenizerTables(str(fx["ckpt"])).id_to_bytes)
    assert R.config_vocab_size(base) == F.REAL_VOCAB_SIZE
    assert not any(s["type"] in (R.SCHEMA_MASKS,) for s in base["sections"])
    assert sum(s["type"] == R.UNICODE_TABLES for s in base["sections"]) == 1, "TOKENIZER without UNICODE_TABLES"
    only = R.read_container(fx["model_only"].read_bytes())
    assert not any(s["type"] in (R.TOKENIZER, R.UNICODE_TABLES) for s in only["sections"])
    assert [s for s in only["sections"]] == [s for s in base["sections"]
                                             if s["type"] not in (R.TOKENIZER, R.UNICODE_TABLES)]
    with_scm1 = R.read_container(fx["with_scm1"].read_bytes())
    scm = [s["data"] for s in with_scm1["sections"] if s["type"] == R.SCHEMA_MASKS]
    assert len(scm) == 1 and set(R.parse_scm1(scm[0])["schemas"]) == {b"g5_minimal_one_field"}
    bad = R.read_container(fx["bad_sil1"].read_bytes())
    diff = [(a["type"]) for a, b in zip(base["sections"], bad["sections"]) if a != b]
    assert diff == [R.SIGMOID_LUT]
    assert R.fnv1a64(R.ZERO_HASH_NAME) == 0 and R.fnv1a64(b"intents") != 0


VERIFIERS = [F.ROOT / "build" / "sslm_verify", F.ROOT / "build" / "Release" / "sslm_verify",
             F.ROOT / "build" / "sslm_verify.exe", F.ROOT / "build" / "Release" / "sslm_verify.exe"]


def test_reference_base_fixtures_are_accepted_by_the_verifier(built, tmp_path):
    """Reference control: the base and every fixture meant to differ only in one named way are accepted by the
    independent verifier (C++ loader), so a refusal of them is the command's, not the verifier's."""
    verifier = next((v for v in VERIFIERS if v.is_file()), None)
    if verifier is None:
        pytest.skip("sslm_verify not built (cmake --build build --target sslm_verify)")
    fx = built()
    for name in ("base", "with_scm1"):
        result = subprocess.run([str(verifier), str(fx[name]), str(tmp_path / f"{name}.manifest.json")],
                                capture_output=True, timeout=600)
        assert result.returncode == 0, f"{name}: {result.stdout.decode()[-400:]} {result.stderr.decode()[-400:]}"


def run(base, ckpt, schemas, out, *extra):
    args = [sys.executable, str(TOOL), "--base", str(base), "--ckpt", str(ckpt)]
    for name, path in schemas:
        args += [b"--schema", (name if isinstance(name, bytes) else name.encode()) + b":" + str(path).encode()]
    args += ["--out", str(out), *map(str, extra)]
    args = [a if isinstance(a, bytes) else a.encode() for a in args]
    return subprocess.run(args, capture_output=True, timeout=1800)


def refused(result, out, *words):
    stderr = result.stderr.decode("utf-8", "replace").lower()
    assert result.returncode != 0, "the command reported success"
    assert any(w in stderr for w in words), f"refusal does not name {words}: {stderr[-400:]}"
    assert not out.exists(), "a refusal left a partial artifact at the output path"


def test_attach_intentdoc_v2_preserves_base_and_reconciles(fx, tmp_path, real_vocab):
    """§4.1: compile against undecoded token bytes with every special id excluded, serialize named MaskPages
    to SCM1, retain every other section and feature flag. The SCM1 is read by an independent parser and
    reconciled with count model v2 at the pinned vocabulary (the must-admit cell through the attach path)."""
    fx = fx()
    vocab, pieces = real_vocab
    out = tmp_path / "attached.sslm"
    result = run(fx["base"], fx["ckpt"], [("intents", fx["intentdoc"])], out)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")[-800:]
    base = R.read_container(fx["base"].read_bytes())
    got = R.read_container(out.read_bytes())
    assert got["flags"] == base["flags"] and got["version"] == base["version"]
    kept = [s for s in got["sections"] if s["type"] != R.SCHEMA_MASKS]
    assert kept == base["sections"], "a pre-existing section or its descriptor changed"
    scm = [s for s in got["sections"] if s["type"] == R.SCHEMA_MASKS]
    assert len(scm) == 1
    parsed = R.parse_scm1(scm[0]["data"])
    assert parsed["vocab"] == R.config_vocab_size(got) == F.REAL_VOCAB_SIZE
    assert set(parsed["schemas"]) == {b"intents"}
    entry = parsed["schemas"][b"intents"]
    model = cm.count(F.INTENTDOC, "intents", pieces, F.REAL_VOCAB_SIZE)
    assert (entry["states"], entry["transitions"], len(entry["accepting"])) == (
        model["dfa_states"], model["token_transitions"], model["accepting"])
    assert len(scm[0]["data"]) == model["section_bytes"]
    assert not any(set(row) & vocab.special for row in entry["rows"].values()), "a special id is admitted"


def _walk_rows(entry, ids):
    state = 0
    for tid in ids:
        if tid not in entry["rows"][state]:
            return False, state
        state = entry["rows"][state][tid]
    return True, state


def test_attached_scm1_walks_as_intentdoc_v2(fx, tmp_path, real_vocab):
    """S0a fold 5 (§6 row 6 "wrong state-offset or accepting-state index"; §7): through the independently
    parsed rows (start state 0, the parsed accepting set), every named document tokenized one byte per
    pinned-vocabulary id, `eight_intents_eight_atoms_each` among them, gets Oracle B's verdict, and the 21
    real-id boundary cases hold. A serializer that shifts accepting ids or permutes CSR offsets keeps every
    count and passes the reconciliation cell; it fails here."""
    fx = fx()
    vocab, _ = real_vocab
    out = tmp_path / "walk.sslm"
    result = run(fx["base"], fx["ckpt"], [("intents", fx["intentdoc"])], out)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")[-800:]
    scm = next(s["data"] for s in R.read_container(out.read_bytes())["sections"] if s["type"] == R.SCHEMA_MASKS)
    entry = R.parse_scm1(scm)["schemas"][b"intents"]
    assert any(name == "eight_intents_eight_atoms_each" for name, _ in C.ACCEPT)
    for name, raw in C.ACCEPT + C.REJECT:
        ok, state = _walk_rows(entry, vocab.bytewise(raw))
        assert (ok and state in entry["accepting"]) == B.accepts(F.INTENTDOC, raw), name
    for name, raw, offset, piece, verdict in T.CASES:
        ok, state = _walk_rows(entry, vocab.bytewise(raw[:offset]))
        assert ok, f"{name}: prefix refused"
        tid = vocab.id_of(piece)
        if verdict:
            ok, end = _walk_rows(entry, vocab.ids(T.pieces(raw, offset, piece)))
            assert ok and end in entry["accepting"], f"{name}: boundary token {piece!r} path refused"
        else:
            assert tid not in entry["rows"][state], f"{name}: opening token {piece!r} admitted"
            assert vocab.id_of(piece[:1]) in entry["rows"][state], f"{name}: one-byte control masked"


def test_attach_repeatable_schemas(fx, tmp_path, real_vocab):
    fx = fx()
    _, pieces = real_vocab
    out = tmp_path / "two.sslm"
    result = run(fx["base"], fx["ckpt"], [("intents", fx["intentdoc"]), ("intent_gate", fx["g1"])], out)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")[-800:]
    parsed = R.parse_scm1(next(s["data"] for s in R.read_container(out.read_bytes())["sections"]
                               if s["type"] == R.SCHEMA_MASKS))
    assert set(parsed["schemas"]) == {b"intents", b"intent_gate"}
    g1 = cm.count(F.G1, "intent_gate", pieces, F.REAL_VOCAB_SIZE)
    assert parsed["schemas"][b"intent_gate"]["transitions"] == g1["token_transitions"]


def test_same_size_wrong_tokenizer_refused(fx, tmp_path):
    fx = fx()
    out = tmp_path / "wrong.sslm"
    refused(run(fx["base"], fx["wrong_ckpt"], [("intents", fx["intentdoc"])], out), out, "tokenizer")


def test_model_only_artifact_refused(fx, tmp_path):
    """Equal vocabulary size alone does not establish token-ID identity (§4.1)."""
    fx = fx()
    out = tmp_path / "model-only.sslm"
    refused(run(fx["model_only"], fx["ckpt"], [("intents", fx["intentdoc"])], out), out, "tokenizer")


def test_duplicate_schema_names_refused(fx, tmp_path):
    fx = fx()
    out = tmp_path / "dup.sslm"
    refused(run(fx["base"], fx["ckpt"], [("intents", fx["intentdoc"]), ("intents", fx["g1"])], out), out, "name")


@pytest.mark.parametrize("name", (R.ZERO_HASH_NAME, b""), ids=("fnv1a64_zero", "empty"))
def test_zero_hash_or_empty_schema_name_refused(fx, tmp_path, name):
    fx = fx()
    out = tmp_path / "zero.sslm"
    refused(run(fx["base"], fx["ckpt"], [(name, fx["intentdoc"])], out), out, "name", "hash")


def test_existing_scm1_replacement_without_merge_policy_refused(fx, tmp_path):
    fx = fx()
    out = tmp_path / "replace.sslm"
    refused(run(fx["with_scm1"], fx["ckpt"], [("intents", fx["intentdoc"])], out), out, "scm1", "schema_masks",
            "merge", "existing")


def test_verifier_runs_before_success(fx, tmp_path):
    """A base whose SigmoidLut payload is damaged (container hash recomputed, so only a content verifier can
    see it): the command must not report success, because the independent verifier runs first."""
    fx = fx()
    out = tmp_path / "unverified.sslm"
    refused(run(fx["bad_sil1"], fx["ckpt"], [("intents", fx["intentdoc"])], out), out, "verif")


@pytest.mark.parametrize("name", tuple(I.PAIRS))
def test_resource_refusal_through_attach_leaves_no_artifact(fx, tmp_path, name, real_vocab):
    """The attach path compiles at the pinned vocabulary, so the limits are derived there (I.pair_pinned)."""
    fx = fx()
    under, over, key, limit = I.pair_pinned(name, real_vocab[1])
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({key: limit}), encoding="utf-8")
    u, o = tmp_path / "under.json", tmp_path / "over.json"
    u.write_text(json.dumps(under), encoding="utf-8")
    o.write_text(json.dumps(over), encoding="utf-8")
    healthy = tmp_path / "healthy.sslm"
    good = run(fx["base"], fx["ckpt"], [("intents", u)], healthy, "--limits-json", limits)
    assert good.returncode == 0 and healthy.exists(), good.stderr.decode("utf-8", "replace")[-800:]
    out = tmp_path / "over.sslm"
    refused(run(fx["base"], fx["ckpt"], [("intents", o)], out, "--limits-json", limits), out,
            F.DIAGNOSTICS[key].lower())


KEYWORD_CASES = [("intents", "maxItems"), ("conditions", "maxItems")] + [
    (location, "vendorUnknownConstraint") for location in F.NODES]


@pytest.mark.parametrize("location,keyword", KEYWORD_CASES, ids=[f"{l}-{k}" for l, k in KEYWORD_CASES])
def test_keyword_refusal_through_attach_leaves_no_artifact(fx, tmp_path, location, keyword):
    fx = fx()
    from keyword_catalogue import VALUES
    schema = F.fresh()
    F.node(schema, location)[keyword] = 4 if keyword == "maxItems" else VALUES[keyword]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    out = tmp_path / "bad.sslm"
    refused(run(fx["base"], fx["ckpt"], [("intents", path)], out), out, keyword.lower())


def test_command_is_documented():
    """§4.1: the command and docs make the feature usable without hand-editing a test fixture."""
    assert TOOL.is_file(), RED
    docs = " ".join(p.read_text(encoding="utf-8", errors="replace") for p in (F.ROOT / "docs").glob("*.md"))
    assert "sslm_attach_schema" in docs and "--schema" in docs, "no document names the command and its inputs"
