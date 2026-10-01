"""Write deterministic, retrieval-stratified Qwen chat prompts for T-3035.

The generated UTF-8 text is committed beside its model-tokenized IDs. The
prefix ends at the assistant header; the model's next token starts an answer.
"""

from pathlib import Path
import argparse
import hashlib


FIRST = ("Mira", "Jules", "Nadia", "Orin", "Tessa", "Kian", "Lena", "Pavel", "Iris", "Noel", "Rina", "Dara")
LAST = ("Tolsen", "Varek", "Mendis", "Calder", "Sevrin", "Jansen", "Novak", "Marin", "Perrin", "Doran")
ITEM = ("amber tea", "silver thread", "blue lantern", "cedar oil", "linen map", "copper bell", "mint cordial", "river stone")
SYSTEM = ("<|im_start|>system\nYou are a meticulous record clerk. Copy the requested records in order, "
          "including the order code, courier, and item. Continue through the comparison and closing recap; "
          "do not abbreviate or end after the first records.\n<|im_end|>\n")


def fact(family: int, i: int) -> str:
    digest = hashlib.sha256(f"T3035:{family}:{i}".encode()).digest()
    code = 10000 + int.from_bytes(digest[:2], "little") % 80000
    courier = FIRST[digest[2] % len(FIRST)] + " " + LAST[digest[3] % len(LAST)]
    item = ITEM[digest[4] % len(ITEM)]
    return f"Record {i:03d}: order K-{code} goes by courier {courier}; its item is {item}.\n"


def query(indices: list[int]) -> str:
    numbers = ", ".join(f"{i:03d}" for i in indices)
    return (f"Recite records {numbers}, in that order. For each record, give its number, complete order code, "
            "full courier name, and item in a separate sentence. Then compare the first and last requested "
            "orders in at least four complete sentences, using their exact details. Finish with a numbered "
            "recap of every requested courier and item.\n")


def indices(n: int) -> list[int]:
    points = [0, 1, n // 4, n // 2 - 1, n // 2, 3 * n // 4, n - 2, n - 1]
    return sorted(set(x for x in points if 0 <= x < n))


def write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))
    print(path.name, len(text.encode("utf-8")), hashlib.sha256(text.encode("utf-8")).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    flat_counts = (2, 4, 8, 12, 21, 30, 37, 48, 59, 66)
    for j, n in enumerate(flat_counts):
        facts = "".join(fact(j, i) for i in range(n))
        user = f"These are {n} independent dispatch records.\n" + facts + query(indices(n))
        system = ("<|im_start|>system\nCopy each requested record precisely. Give a long answer.\n<|im_end|>\n"
                  if j == 0 else SYSTEM)
        text = system + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
        write(args.directory / f"flat{j:02d}.txt", text)

    world_count = 34
    world = SYSTEM + "<|im_start|>user\nHere is the shared dispatch ledger.\n"
    world += "".join(fact(100, i) for i in range(world_count))
    world += "Remember these records for the next user turn.\n<|im_end|>\n"
    write(args.directory / "world.txt", world)
    for j in range(10):
        persona_count = 2 + (j % 2)
        suffix = "<|im_start|>user\nHere is the persona's additional ledger.\n"
        suffix += "".join(fact(200 + j, world_count + i) for i in range(persona_count))
        requested = [0, 1, 9, 18, 27, 35, 36, world_count, world_count + persona_count // 2,
                     world_count + persona_count - 1]
        suffix += query(requested) + "<|im_end|>\n<|im_start|>assistant\n"
        write(args.directory / f"persona{j:02d}.txt", suffix)
        write(args.directory / f"nested{j:02d}.txt", world + suffix)


if __name__ == "__main__":
    main()
