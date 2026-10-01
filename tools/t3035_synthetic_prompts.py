"""Pin T-3035 synthetic Stream() IDs at the real prompt's prefix lengths."""

from pathlib import Path
import argparse
import hashlib


def stream(seed: int, length: int) -> list[int]:
    x = (seed * 2654435761 + 12345) & 0xFFFFFFFF
    out = []
    for _ in range(length):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        out.append((x >> 8) % 256)
    return out


def count(path: Path) -> int:
    return len(path.read_text(encoding="ascii").split())


def write(directory: Path, name: str, description: str, ids: list[int]) -> None:
    text = description.encode("utf-8")
    raw_ids = (" ".join(str(x) for x in ids) + "\n").encode("ascii")
    (directory / f"{name}.txt").write_bytes(text)
    (directory / f"{name}.ids").write_bytes(raw_ids)
    print(name, len(ids), hashlib.sha256(text).hexdigest(), hashlib.sha256(raw_ids).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("real_prompt_directory", type=Path)
    parser.add_argument("synthetic_prompt_directory", type=Path)
    args = parser.parse_args()
    args.synthetic_prompt_directory.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        name = f"flat{i:02d}"
        length = count(args.real_prompt_directory / f"{name}.ids")
        seed = 51 + i * 7
        write(args.synthetic_prompt_directory, name,
              f"T-3035 synthetic Stream(seed={seed}, length={length}, vocabulary=256).\n",
              stream(seed, length))
    world_length = count(args.real_prompt_directory / "world.ids")
    world = stream(61, world_length)
    write(args.synthetic_prompt_directory, "world",
          f"T-3035 synthetic shared world Stream(seed=61, length={world_length}, vocabulary=256).\n", world)
    for i in range(10):
        persona_name = f"persona{i:02d}"
        persona_length = count(args.real_prompt_directory / f"{persona_name}.ids")
        seed = 71 + i
        persona = stream(seed, persona_length)
        description = f"T-3035 synthetic persona Stream(seed={seed}, length={persona_length}, vocabulary=256).\n"
        write(args.synthetic_prompt_directory, persona_name, description, persona)
        nested_name = f"nested{i:02d}"
        write(args.synthetic_prompt_directory, nested_name,
              f"T-3035 synthetic nested world=61/{world_length}; persona={seed}/{persona_length}.\n",
              world + persona)
    donor_length = max(count(args.real_prompt_directory / f"flat{i:02d}.ids") for i in range(10)) + 128
    write(args.synthetic_prompt_directory, "donor",
          f"T-3035 synthetic donor Stream(seed=997, length={donor_length}, vocabulary=256).\n",
          stream(997, donor_length))


if __name__ == "__main__":
    main()
