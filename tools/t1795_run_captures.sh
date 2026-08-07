#!/bin/bash
# T-1795: run the engine-side residual probe and the float-side residual reference on this
# ticket's 3 held-out prompts, serially. Prompt strings are built with $'...' literals and
# passed as argv directly -- never through $(...) command substitution, which strips the
# chat template's trailing newline (T-1788's own tokenization-parity gate).
set -e
cd "$(dirname "$0")/.."

MODEL="D:/hf_cache/superslm_artifacts/qwen2.5-1.5b-instruct.sslm"
TOK="tests/fixtures/qwen2.5-1.5b.tok.sslm"
SYS="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

U1="What is a good pillow choice for someone with back pain?"
U2="How does a compass work?"
U3="Write a short thank-you note after a job interview."

mkdir -p out/t1795

i=1
for U in "$U1" "$U2" "$U3"; do
  P=$'<|im_start|>system\n'"$SYS"$'<|im_end|>\n<|im_start|>user\n'"$U"$'<|im_end|>\n<|im_start|>assistant\n'
  echo "=== engine probe P$i ==="
  ./out/t1795_residual_probe.exe "$MODEL" "$TOK" "$P" "p$i" --dump-dir "out/t1795" \
    > "out/t1795/p${i}_run.log" 2>&1
  tail -5 "out/t1795/p${i}_run.log"
  i=$((i+1))
done

i=1
for U in "$U1" "$U2" "$U3"; do
  echo "=== float reference P$i ==="
  python tools/t1795_float_residual_reference.py "$U" --system "$SYS" \
    --dump "out/t1795/p${i}_float.txt" \
    > "out/t1795/p${i}_float.log" 2>&1
  tail -8 "out/t1795/p${i}_float.log"
  i=$((i+1))
done

echo "=== all captures done ==="

i=1
for U in "$U1" "$U2" "$U3"; do
  echo "=== float attn_out dump P$i ==="
  python tools/t1795_float_attnout_dump.py "$U" --system "$SYS" --dump "out/t1795/p${i}_float_attnout.txt" \
    > "out/t1795/p${i}_float_attnout.log" 2>&1
  tail -3 "out/t1795/p${i}_float_attnout.log"
  i=$((i+1))
done
