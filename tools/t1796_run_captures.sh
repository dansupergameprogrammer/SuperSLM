#!/bin/bash
# T-1796: run the boundary-proximity probe on 3 held-out prompts, serially. Topics distinct
# from every prior prompt set in this campaign (T-1762/T-1763's dinner-party/stack-queue set,
# T-1795's pillow/compass/thank-you-note set, and T-1786/T-1787's own held-out sets).
set -e
cd "$(dirname "$0")/.."

MODEL="D:/hf_cache/superslm_artifacts/qwen2.5-1.5b-instruct.sslm"
TOK="tests/fixtures/qwen2.5-1.5b.tok.sslm"
SYS="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

U1="Explain why the sky is blue in simple terms."
U2="Suggest three exercises for someone just starting to run."
U3="What should I consider when choosing a router for my apartment?"

mkdir -p out/t1796

i=1
for U in "$U1" "$U2" "$U3"; do
  P=$'<|im_start|>system\n'"$SYS"$'<|im_end|>\n<|im_start|>user\n'"$U"$'<|im_end|>\n<|im_start|>assistant\n'
  echo "=== boundary probe P$i ==="
  ./out/t1796_boundary_probe.exe "$MODEL" "$TOK" "$P" "p$i" \
    > "out/t1796/p${i}_run.log" 2>&1
  tail -10 "out/t1796/p${i}_run.log"
  i=$((i+1))
done

echo "=== all captures done ==="
