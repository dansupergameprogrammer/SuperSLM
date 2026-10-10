"""Commissioning cells for check_gpu_timing_mode_abi_unchanged.py (GPU per-token overhead plan
rev 3.1, cell 7.9): green on this tree, green on a CRLF copy of the same bytes, red on a one-byte
change to either pinned file."""
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_gpu_timing_mode_abi_unchanged as check  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _copy_tree(tmp_path):
    for rel in check.PINS:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    return tmp_path


def test_green_on_this_tree():
    assert check.main(["--root", str(ROOT)]) == 0


def test_green_on_crlf_checkout(tmp_path):
    root = _copy_tree(tmp_path)
    for rel in check.PINS:
        p = root / rel
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert check.main(["--root", str(root)]) == 0


def test_red_on_a_changed_byte(tmp_path):
    for rel in check.PINS:
        root = _copy_tree(tmp_path / rel.replace("/", "_"))
        p = root / rel
        p.write_bytes(p.read_bytes() + b"// added\n")
        assert check.main(["--root", str(root)]) == 1
