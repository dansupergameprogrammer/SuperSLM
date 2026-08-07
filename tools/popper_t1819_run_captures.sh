#!/bin/bash
# T-1819: re-run T-1795's Phase A/B capture on THREE PROMPTS POPPER CHOSE, topics distinct
# from every prompt set T-1795, T-1796, and the tickets they cite used (pillow/compass/
# thank-you-note; camping/tides/birthday; icebreaker/rainbow/new-job; dinner-party/
# stack-queue; sky-blue/running-exercises/router).  Same binary, same model, same tokenizer,
# same float reference script -- only the inputs change.  This is the overfit attack on
# claims 1 and 2.
#
# Prompt strings are $'...' literals passed as argv, never through $(...) -- command
# substitution strips the chat template's trailing newline.
set -e
cd "$(dirname "$0")/.."

MODEL="D:/hf_cache/superslm_artifacts/qwen2.5-1.5b-instruct.sslm"
TOK="tests/fixtures/qwen2.5-1.5b.tok.sslm"
SYS="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

U1="How do I keep houseplants alive in a low-light apartment?"
U2="What is the difference between a virus and a bacterium?"
U3="Give me a packing list for a weekend cycling trip in cold weather, and explain why each item matters."

mkdir -p out/popper_t1819

i=1
for U in "$U1" "$U2" "$U3"; do
  P=$'<|im_start|>system\n'"$SYS"$'<|im_end|>\n<|im_start|>user\n'"$U"$'<|im_end|>\n<|im_start|>assistant\n'
  echo "=== engine probe Q$i ==="
  ./out/t1795_residual_probe.exe "$MODEL" "$TOK" "$P" "q$i" --dump-dir "out/popper_t1819" \
    > "out/popper_t1819/q${i}_run.log" 2>&1
  tail -4 "out/popper_t1819/q${i}_run.log"
  i=$((i+1))
done

i=1
for U in "$U1" "$U2" "$U3"; do
  echo "=== float reference Q$i ==="
  python tools/t1795_float_residual_reference.py "$U" --system "$SYS" \
    --dump "out/popper_t1819/q${i}_float.txt" > "out/popper_t1819/q${i}_float.log" 2>&1
  tail -6 "out/popper_t1819/q${i}_float.log"
  i=$((i+1))
done

echo "=== popper captures done ==="
