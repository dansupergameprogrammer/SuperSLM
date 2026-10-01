"""Commission the independent SPX1 field oracle without prefix ABI verbs."""

from argparse import Namespace
from hashlib import sha256
from pathlib import Path
from struct import pack_into
from tempfile import TemporaryDirectory

from check_spx1 import check, fnv1a64


def put(data: bytearray, fmt: str, offset: int, value: int) -> None:
    pack_into("<" + fmt, data, offset, value)


def digest(data: bytearray) -> None:
    data[28:60] = bytes(32)
    data[28:60] = sha256(data).digest()


def main() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        args = Namespace(blob=root / "prefix.spx1", ssb=root / "adopter.ssb6",
                         rows=root / "rows.bin", cmake=root / "CMakeLists.txt",
                         model_sha="55" * 32, current_token=23,
                         schema_name="schema_two", walk=1,
                         counts="11,13,17,19,23", length=1, origin=0,
                         layers=1, heads=1, dim=1, hidden=192)
        args.cmake.write_text("project(superslm VERSION 1.2.3)\n", encoding="utf-8")
        args.rows.write_bytes(b"\x33\x44")
        blob = bytearray(184 + 192 + 2)
        ssb = bytearray(172 + 192)
        blob[:4], ssb[:4] = b"SPX1", b"SSB6"
        blob[60:92] = ssb[4:36] = bytes.fromhex(args.model_sha)
        for offset, value in ((8, 1), (12, 2), (16, 3), (92, 1)):
            put(blob, "I", offset, value)
        put(ssb, "I", 36, 1)
        for blob_offset, ssb_offset, fmt, value in (
            (96, 40, "Q", fnv1a64(args.schema_name)),
            (104, 48, "I", 1), (112, 60, "q", 1),
            (128, 76, "q", 17), (136, 84, "q", -3),
            (144, 92, "Q", 11), (152, 124, "Q", 13),
            (160, 132, "Q", 17), (168, 140, "Q", 19),
            (176, 148, "Q", 23),
        ):
            put(blob, fmt, blob_offset, value)
            put(ssb, fmt, ssb_offset, value)
        put(blob, "i", 108, 23)
        put(ssb, "i", 72, -1)  # Ready-for-logits SSB6 has no pending embed.
        blob[184:376] = ssb[172:364] = bytes(range(192))
        blob[376:] = args.rows.read_bytes()

        def accepts(candidate: bytearray, seq: bytearray) -> None:
            digest(candidate)
            args.blob.write_bytes(candidate)
            args.ssb.write_bytes(seq)
            check(args)

        def rejects(candidate: bytearray, seq: bytearray, reason: str) -> None:
            digest(candidate)
            args.blob.write_bytes(candidate)
            args.ssb.write_bytes(seq)
            try:
                check(args)
            except AssertionError as error:
                assert reason in str(error), (reason, str(error))
            else:
                raise AssertionError(f"oracle accepted {reason} mutation")

        accepts(blob.copy(), ssb.copy())
        altered = blob.copy()
        put(altered, "i", 108, 24)
        rejects(altered, ssb.copy(), "known last prefill token")
        altered = blob.copy()
        altered_seq = ssb.copy()
        put(altered, "Q", 96, fnv1a64("schema_one"))
        put(altered_seq, "Q", 40, fnv1a64("schema_one"))
        rejects(altered, altered_seq, "independent bound schema hash")
        altered = blob.copy()
        altered_seq = ssb.copy()
        put(altered, "I", 104, 2)
        put(altered_seq, "I", 48, 2)
        rejects(altered, altered_seq, "known advanced DFA walk")
        altered = blob.copy()
        altered_seq = ssb.copy()
        put(altered, "Q", 152, 17)
        put(altered, "Q", 160, 13)
        put(altered_seq, "Q", 124, 17)
        put(altered_seq, "Q", 132, 13)
        rejects(altered, altered_seq, "known saturation counts")
    print("SPX1 oracle: accepted distinct-count fixture; rejected token, schema, walk, symmetric count swap")


if __name__ == "__main__":
    main()
