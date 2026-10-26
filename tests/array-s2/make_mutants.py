"""S2 mutant check: writes two scratch copies of tools/sslm_attach_schema.py, each patched to
corrupt the `intents` schema's SCM1 bytes right after `serialize_scm1` (the mutations are
scm1_mutants.py's), so the verifier, the walks and the generation of tests/array-s2 can be run
against a bad artifact.

    python make_mutants.py <repo tools dir> <scratch dir>

Run each copy with the repo's tools directory on PYTHONPATH (the copy's own directory comes first,
so its patched module is the one imported).
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

MUTATE = '''

def _mutate(scm1):
    import sys as _sys
    _sys.path.insert(0, {here!r})
    import scm1_mutants
    out, what = scm1_mutants.mutate(scm1, MODE)
    print("MUTANT " + what)
    return out
'''


def main() -> int:
    tools, out = Path(sys.argv[1]), Path(sys.argv[2])
    for mode in ("accept_shift", "offset_swap"):
        d = out / mode
        d.mkdir(parents=True, exist_ok=True)
        for f in ("sslm_format.py", "sslm_convert_schema.py"):
            shutil.copy(tools / f, d / f)
        src = (tools / "sslm_attach_schema.py").read_text(encoding="utf-8")
        anchor = "    scm1 = SC.serialize_scm1(compiled, vocab_size)\n"
        assert src.count(anchor) == 1
        src = src.replace(anchor, anchor + "    scm1 = _mutate(scm1)\n")
        src = src.replace("\ndef main(", f"\nMODE = {mode!r}\n" + MUTATE.format(here=str(HERE)) + "\n\ndef main(", 1)
        (d / "sslm_attach_schema.py").write_text(src, encoding="utf-8")
        print("wrote", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
