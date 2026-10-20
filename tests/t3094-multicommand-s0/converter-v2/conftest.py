import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent, HERE.parent / "v2", HERE.parents[2] / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2_fixtures as F  # noqa: E402
import cv2_oracle as B  # noqa: E402


@pytest.fixture(scope="session")
def real_vocab():
    """The pinned Qwen2.5-0.5B-Instruct (snapshot 7ae5576) vocabulary: raw token bytes by id and its special
    ids, loaded by count model v2's own independent loader (not the converter's tokenizer tooling)."""
    tokenizer = F.TOKENIZER_DIR / "tokenizer.json"
    if not tokenizer.is_file():
        pytest.skip(f"pinned tokenizer absent at {tokenizer} (set T3094_TOKENIZER_DIR)")
    import count_model_v2 as cm
    import json
    pieces, _ = cm.load_vocab(tokenizer, F.REAL_VOCAB_SIZE)
    special = [a["id"] for a in json.loads(tokenizer.read_text(encoding="utf-8")).get("added_tokens", [])]
    return B.Vocabulary(pieces, special), pieces
