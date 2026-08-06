#!/bin/bash
# T-1792: run the engine-side probe (all 28 layers) and the float-side reference (all 28 layers)
# on 3 NEW held-out prompts, distinct from every prompt set used earlier in this campaign
# (T-1762/T-1763's dinner-party/stack-queue/apology-email; T-1769's lamp-wiring/hash-map/
# missed-meeting; T-1776's bicycle/vaccine/thank-you; T-1778/T-1784's bread-recipe/
# photosynthesis/declining-meeting; T-1787's houseplants set; T-1789/T-1791's camping-trip/
# ocean-tides/birthday-message). Serially, one model-resident process at a time. Prompt strings
# are built with $'...' literals and passed as argv directly -- never through $(...) command
# substitution, which strips the trailing newline the chat template ends in (T-1788's own
# tokenization-parity gate).
set -e
cd "$(dirname "$0")/.."

MODEL="D:/hf_cache/superslm_artifacts/qwen2.5-1.5b-instruct.sslm"
TOK="tests/fixtures/qwen2.5-1.5b.tok.sslm"
SYS="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

U1="What are some good icebreaker questions for a work team meeting?"
U2="Explain how a rainbow forms in simple terms."
U3="Write a short congratulations note for a friend who just started a new job."

mkdir -p out/t1792

i=1
for U in "$U1" "$U2" "$U3"; do
  P=$'<|im_start|>system\n'"$SYS"$'<|im_end|>\n<|im_start|>user\n'"$U"$'<|im_end|>\n<|im_start|>assistant\n'
  echo "=== engine probe P$i ==="
  ./out/t1792_attn_probe.exe "$MODEL" "$TOK" "$P" --dump-prefix "out/t1792/p$i" \
    > "out/t1792/p${i}_run.log" 2>&1
  tail -3 "out/t1792/p${i}_run.log"
  i=$((i+1))
done

i=1
for U in "$U1" "$U2" "$U3"; do
  echo "=== float reference P$i ==="
  python tools/t1792_float_attn_reference.py "$U" --system "$SYS" \
    --dump "out/t1792/p${i}_float.txt" \
    > "out/t1792/p${i}_float.log" 2>&1
  tail -5 "out/t1792/p${i}_float.log"
  i=$((i+1))
done

echo "=== all captures done ==="
