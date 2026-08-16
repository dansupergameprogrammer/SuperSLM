"""t2139_build_combined_fixture.py -- T-2139 (Brunel): produces the real combined .sslm
(weights + tokenizer, one file, one sslm_model_map call) the S-FREEZE example and C7's own
round-trip need. Not a new converter: composes convert_model.py's own build_sections() and
convert_tokenizer.py's own TokenizerTables (the project's real, already-proven section-writer
functions, sslm_format.py's shared SectionType/Section/write_artifact) into ONE sections list,
then calls the SAME write_artifact() both existing CLIs already call. Never hand-crafts a byte.

Design finding this script exists to route around correctly (Claude/Brunel/
t2139-abi-build-2026-08-16.md, this fold): convert_model.py's own emitted artifact and
convert_tokenizer.py's own emitted artifact are each a complete, independent container -- and
convert_tokenizer.py's own emit_artifact() writes ITS OWN CONFIG section (a JSON blob under
SectionType.CONFIG) and its own SIGMOID_LUT section, both under the SAME SectionType values
convert_model.py's own CFG1 Config and canonical SigmoidLut already occupy. A real .sslm carries
at most one section per SectionType (SslmModelView::Section(type) returns one pointer, never a
list) -- combining both tools' FULL section lists verbatim would collide on two types. This
script takes convert_model.py's own real Config/Weights/.../SigmoidLut sections UNCHANGED, and
ONLY convert_tokenizer.py's own Tokenizer/UnicodeTables/ChatTemplate sections (dropping its
redundant Config/SigmoidLut, which convert_model.py's own sections already supply correctly).

Usage: python t2139_build_combined_fixture.py --artifact <calibrated-dir> --ckpt <hf-ckpt-dir>
                                               --out <combined.sslm>
"""
import argparse
import sys

import sslm_convert_validate as V
import sslm_format as F
import convert_model as CM
import convert_tokenizer as CT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="calibrated artifact directory (calibrate_checkpoint.py's own --out)")
    ap.add_argument("--ckpt", required=True, help="HF checkpoint dir with tokenizer.json (convert_tokenizer.py's own --ckpt)")
    ap.add_argument("--out", required=True, help="output combined .sslm path")
    args = ap.parse_args()

    artifact_cache, _pipeline = CM._load_spike()
    model = artifact_cache.load_artifact(args.artifact)

    # Phase 1: validate -- the SAME call convert_model.py's own main() makes, unmodified.
    V.validate_model(model, fold_ops_tensor=CM._fold_ops_tensor, ctx_fold_tensor=CM._ctx_fold_tensor,
                      unicode_major=V.PINNED_UNICODE_VERSION[0],
                      unicode_minor=V.PINNED_UNICODE_VERSION[1],
                      unicode_patch=V.PINNED_UNICODE_VERSION[2])

    # Phase 2: the real weight sections -- convert_model.py's own build_sections(), unmodified.
    model_sections, fold_approx_error = CM.build_sections(model)

    # Phase 3: the real tokenizer sections -- convert_tokenizer.py's own TokenizerTables, taking
    # only TOKENIZER/UNICODE_TABLES/CHAT_TEMPLATE (its own CONFIG/SIGMOID_LUT sections are
    # dropped, per this file's own header comment -- convert_model.py's own sections already
    # supply the real CFG1 Config and the real canonical SigmoidLut for those two types).
    tables = CT.TokenizerTables(args.ckpt)
    tokenizer_sections = [
        F.Section(F.SectionType.TOKENIZER, tables.serialize_tokenizer()),
        F.Section(F.SectionType.UNICODE_TABLES, tables.u.serialize()),
        F.Section(F.SectionType.CHAT_TEMPLATE,
                  __import__("json").dumps({"chat_template": tables.chat_template}, sort_keys=True).encode("utf-8")),
    ]

    combined = model_sections + tokenizer_sections
    seen_types = [s.type for s in combined]
    if len(seen_types) != len(set(seen_types)):
        dupes = sorted({t for t in seen_types if seen_types.count(t) > 1})
        print(f"FAIL: duplicate section types in combined list: {dupes}", file=sys.stderr)
        return 1

    fp = F.write_artifact(args.out, combined)
    print(f"wrote {args.out}")
    print(f"fingerprint {fp}")
    print(f"sections {len(combined)}: " + ", ".join(str(s.type) for s in combined))
    print(f"fold_approximation_error {fold_approx_error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
