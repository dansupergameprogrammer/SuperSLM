#!/bin/bash
# T-1820: capture the engine's and the float32 reference's residual stream at EVERY prompt
# position, on the SAME three prompts T-1795 used (so this ticket's last-position rows can be
# checked against T-1795's own published table) plus the three T-1819 chose independently
# (so nothing here is measured only on the set the original finding was found on).
#
# Runs SERIALLY -- one process at a time, cap 4, used 1.
#
# Prompt strings are $'...' literals passed as argv, never through $(...) -- command
# substitution strips the chat template's trailing newline (T-1788's tokenization-parity gate).
set -e
cd "$(dirname "$0")/.."

MODEL="D:/hf_cache/superslm_artifacts/qwen2.5-1.5b-instruct.sslm"
TOK="tests/fixtures/qwen2.5-1.5b.tok.sslm"
SYS="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# T-1795's own three prompts (p1..p3), then T-1819's own three (q1..q3).
IDS=(p1 p2 p3 q1 q2 q3)
USERS=(
  "What is a good pillow choice for someone with back pain?"
  "How does a compass work?"
  "Write a short thank-you note after a job interview."
  "How do I keep houseplants alive in a low-light apartment?"
  "What is the difference between a virus and a bacterium?"
  "Give me a packing list for a weekend cycling trip in cold weather, and explain why each item matters."
)

mkdir -p out/t1820

for i in "${!IDS[@]}"; do
  ID="${IDS[$i]}"
  U="${USERS[$i]}"
  P=$'<|im_start|>system\n'"$SYS"$'<|im_end|>\n<|im_start|>user\n'"$U"$'<|im_end|>\n<|im_start|>assistant\n'
  echo "=== engine probe $ID ==="
  ./out/t1820_position_residual_probe.exe "$MODEL" "$TOK" "$P" "$ID" --dump-dir "out/t1820" \
    > "out/t1820/${ID}_engine.log" 2>&1
  tail -3 "out/t1820/${ID}_engine.log"
done

for i in "${!IDS[@]}"; do
  ID="${IDS[$i]}"
  U="${USERS[$i]}"
  echo "=== float reference $ID ==="
  python -u tools/t1820_float_position_reference.py "$U" --system "$SYS" \
    --dump "out/t1820/${ID}_float.npy" > "out/t1820/${ID}_float.log" 2>&1
  tail -4 "out/t1820/${ID}_float.log"
done

echo "=== all captures done ==="
