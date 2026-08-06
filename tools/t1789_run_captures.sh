#!/bin/bash
# T-1789: run the engine-side probe and the float-side reference on the ticket's 3 held-out
# prompts, serially (one model-resident process at a time, per the campaign's own discipline).
# Prompt strings are built with $'...' literals and passed as argv directly -- NEVER through
# $(...) command substitution, which strips the trailing newline the chat template ends in
# (T-1788's own tokenization-parity gate, discovered there, applied here from the start).
set -e
cd "$(dirname "$0")/.."

MODEL="D:/hf_cache/superslm_artifacts/qwen2.5-1.5b-instruct.sslm"
TOK="tests/fixtures/qwen2.5-1.5b.tok.sslm"
SYS="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

U1="What should I pack for a weekend camping trip in the mountains?"
U2="How do ocean tides work?"
U3="Write a short birthday message for my grandmother."

mkdir -p out/t1789

i=1
for U in "$U1" "$U2" "$U3"; do
  P=$'<|im_start|>system\n'"$SYS"$'<|im_end|>\n<|im_start|>user\n'"$U"$'<|im_end|>\n<|im_start|>assistant\n'
  EXTRA=""
  if [ "$i" = "1" ]; then EXTRA="--weights-dump out/t1789/weights.bin"; fi
  echo "=== engine probe P$i ==="
  ./out/t1789_attention_probe.exe "$MODEL" "$TOK" "$P" --dump-prefix "out/t1789/p$i" $EXTRA \
    > "out/t1789/p${i}_run.log" 2>&1
  tail -3 "out/t1789/p${i}_run.log"
  i=$((i+1))
done

i=1
for U in "$U1" "$U2" "$U3"; do
  echo "=== float reference P$i ==="
  python tools/t1789_float_reference.py "$U" --system "$SYS" \
    --dump "out/t1789/p${i}_float.txt" --hidden-dump "out/t1789/p${i}_float_hidden.txt" \
    > "out/t1789/p${i}_float.log" 2>&1
  tail -5 "out/t1789/p${i}_float.log"
  i=$((i+1))
done

echo "=== all captures done ==="
