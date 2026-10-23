# G1: the two P6 pkv_def references 6.2/E2 and 6.3/E3 read, regenerated at 1948e1f2

Plan rev 1.18.15 §15.35 G1, bound as §15.33 (a) and rev 1.18.13a say. Written by the test author in the cloud on
2026-10-23. **Not licensed.** The debunker licenses next, and the two ctests grade after that.

| Input | Value |
|---|---|
| Generator | `superslm_precache_reference_gen` (`tests/precache/reference/t3035_reference.cpp`), the E4 generator commit `47bdcc9e`, clean detached worktree, Linux GCC 13.3 Release |
| Compiled anchor / harness | `1948e1f2ea4528790dc6701e26c47bda5739b003` / `47bdcc9e24917f91e5f9bf67e8a0141662ca5114` (no `unanchored` line at configure) |
| Model | `pkv_def.sslm` from `tools/gen_paged_kv_fixture.py`, sha256 `994b58b4…` |
| Prompts | `tests/t3035-precache-refs/synthetic-prompts/` at `origin/curie/t3035-precache-refs` `972d5b7b` (the ID hashes equal the 5d5ca58 population's headers) |
| Tokenizer label, ID suffix, count | `synthetic-Stream(e8b3fdd)` (the 5d5ca58 population's label), `ids`, `1` |
| RECORD | `…/discrimination-v2/P6-pkv_def-flat/{id}.stream.json` and `…/P6-pkv_def-nested/{id}.stream.json` at `972d5b7b` |

The commands ran run1 and run2 in separate processes, each flat and then nested:

```
superslm_precache_reference_gen pkv_def.sslm synthetic-prompts OUT/P6-pkv_def-flat   flat   runN 'synthetic-Stream(e8b3fdd)' ids V2/P6-pkv_def-flat/{id}.stream.json   1
superslm_precache_reference_gen pkv_def.sslm synthetic-prompts OUT/P6-pkv_def-nested nested runN 'synthetic-Stream(e8b3fdd)' ids V2/P6-pkv_def-nested/{id}.stream.json 1
```

## Pair check

| Stem | run1 = run2 (sha256) | Header | Bound record (stream-v2) |
|---|---|---|---|
| `P6-pkv_def-flat/flat00` | yes, `18b914b15faaf5631e54f661753d739b9ca3a40b10afc27c3f45940b8625a8f1` | anchor `1948e1f2…`, harness `47bdcc9e…`, B=128, cap=4096, model `994b58b4…` | `flat00.stream.json`, sha256 `8fd44bc7024bb523356a080eac338f81f7037fa0c5d32cba6d0f2c32fcf759b1` |
| `P6-pkv_def-nested/nested00` | yes, `a85c2228ded206296a7f59399eb00db36ebd22dc1a856f8f7911f537c9b82674` | as flat00, plus the `world_*` and `persona_*` hashes | `nested00.stream.json`, sha256 `fbb513a713b16d4e0bc2a465bc359621315fedb24615ab32f02f6998757913d6` |

The same check passed on every one of these points, for both stems:
- The header's `discrimination_record_sha256` equals the sidecar file's SHA-256.
- The sidecar's `instrument` is `T-3035-precache-stream-v2`.
- `prompt_ids_sha256` equals the sidecar's `target_ids_sha256`.
- There are 5 rows of 64 tokens with `terminal_status=0` and `eos_index=-1`.
- The legacy greedy stream (adopter 1) is token-identical to the sidecar's `unedited_roundtrip_tokens`, which was
  measured on the B = 16 population. The B-invariance §15.33 (4) relies on holds here.

**For the debunker:** each sidecar still carries its historical JSON label `QUARANTINED` (T-3071 §1). Standing is
read from the registry: `T-3035-precache-stream-v2` is COMMISSIONED (D-SLM8030). The sidecar was measured at
`page_positions` 16. The licence's `record_hashes` for these two stems are the two stream hashes above. Each stem's
`byte.json` is provenance only, as rev 1.18.13a says.

`SHA256SUMS.txt` holds the four files' hashes. `.gitattributes` keeps them byte-exact.
