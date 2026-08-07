#!/bin/bash
# T-1797 E0: run the engine multiposition probe and the float multiposition reference on
# 15 prompts -- T-1795's own 3 (P1-P3, so E0's population nests the prior measurement) plus
# 12 fresh held-out topics distinct from every prior prompt set in this campaign
# (T-1762/T-1763 dinner-party/stack-queue; T-1786 home-library/compound-interest/
# congratulations-promotion; T-1787 houseplants/weather-climate/buttermilk; T-1789
# camping/tides/birthday; T-1792 icebreaker/rainbow/new-job; T-1795 pillow/compass/
# thank-you-note; T-1796 sky-blue/running/router).
# Prompt strings are $'...' literals passed as argv directly, never through $(...), which
# strips the chat template's trailing newline (T-1788's own tokenization-parity gate).
# Engine runs first (all 15, serial), float second (all 15, serial): heavy model-resident
# forwards are serialized by this campaign's standing rule.
set -e
cd "$(dirname "$0")/.."

MODEL="D:/hf_cache/superslm_artifacts/qwen2.5-1.5b-instruct.sslm"
TOK="tests/fixtures/qwen2.5-1.5b.tok.sslm"
SYS="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

PROMPTS=(
  "What is a good pillow choice for someone with back pain?"
  "How does a compass work?"
  "Write a short thank-you note after a job interview."
  "How do noise-canceling headphones work?"
  "Suggest a simple weekly meal-prep plan for two people."
  "Why do leaves change color in autumn?"
  "Write a short apology email for missing a meeting."
  "What should I look for when buying a used bicycle?"
  "Explain the difference between baking soda and baking powder."
  "How can I improve my posture while working at a desk?"
  "What are good strategies for remembering people's names?"
  "Explain why ice floats on water."
  "Write a two-sentence product description for a ceramic mug."
  "How does yeast make bread rise?"
  "What should I check before a long highway drive?"
)

mkdir -p out/t1797

STAGE="${1:-all}"

if [ "$STAGE" = "engine" ] || [ "$STAGE" = "all" ]; then
  i=1
  for U in "${PROMPTS[@]}"; do
    P=$'<|im_start|>system\n'"$SYS"$'<|im_end|>\n<|im_start|>user\n'"$U"$'<|im_end|>\n<|im_start|>assistant\n'
    echo "=== engine probe p$i ==="
    ./out/t1797_multipos_probe.exe "$MODEL" "$TOK" "$P" "p$i" --dump-dir "out/t1797" \
      > "out/t1797/p${i}_eng.log" 2>&1
    tail -2 "out/t1797/p${i}_eng.log"
    i=$((i+1))
  done
fi

if [ "$STAGE" = "float" ] || [ "$STAGE" = "all" ]; then
  i=1
  for U in "${PROMPTS[@]}"; do
    echo "=== float reference p$i ==="
    python tools/t1797_float_multipos.py "$U" --system "$SYS" \
      --dump "out/t1797/p${i}_float.bin" > "out/t1797/p${i}_float.log" 2>&1
    tail -3 "out/t1797/p${i}_float.log"
    i=$((i+1))
  done
fi

echo "=== E0 captures done ($STAGE) ==="
