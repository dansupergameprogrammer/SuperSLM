#!/bin/bash
# T-1799: run the widened-accumulate-only product candidate (the existing t1797 "k8" arm,
# unmodified) plus its k0 self-check, on 9 held-out prompts.
#
# Prompt set: T-1797 E0's p7-p15 (the 9 fresh-topic prompts never used by the T-1797 solve's
# E1/E2 arms or the T-1798 debunk's nohad ablation, both of which used only p1-p6). Prompt
# strings and $'...' literal construction copied verbatim from tools/t1797_run_e0.sh so the
# tokenization matches the existing out/t1799/p{7..15}_float.bin / p{7..15}_eng.bin exactly
# (T-1788's tokenization-parity gate: command substitution strips the chat template's
# trailing newline and silently changes the token sequence -- avoided here the same way).
set -e
cd "$(dirname "$0")/.."

MODEL="D:/hf_cache/superslm_artifacts/qwen2.5-1.5b-instruct.sslm"
TOK="tests/fixtures/qwen2.5-1.5b.tok.sslm"
SYS="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

declare -A PROMPTS
PROMPTS[7]="Write a short apology email for missing a meeting."
PROMPTS[8]="What should I look for when buying a used bicycle?"
PROMPTS[9]="Explain the difference between baking soda and baking powder."
PROMPTS[10]="How can I improve my posture while working at a desk?"
PROMPTS[11]="What are good strategies for remembering people's names?"
PROMPTS[12]="Explain why ice floats on water."
PROMPTS[13]="Write a two-sentence product description for a ceramic mug."
PROMPTS[14]="How does yeast make bread rise?"
PROMPTS[15]="What should I check before a long highway drive?"

mkdir -p out/t1799arms

for i in 7 8 9 10 11 12 13 14 15; do
  U="${PROMPTS[$i]}"
  P=$'<|im_start|>system\n'"$SYS"$'<|im_end|>\n<|im_start|>user\n'"$U"$'<|im_end|>\n<|im_start|>assistant\n'
  for ARM in k0 k4 k8; do
    echo "=== p$i arm=$ARM ==="
    ./out/t1797_residual_arms.exe "$MODEL" "$TOK" "$P" "p$i" --arm "$ARM" \
      --dump-dir "out/t1799arms" > "out/t1799arms/p${i}_${ARM}.log" 2>&1
    tail -3 "out/t1799arms/p${i}_${ARM}.log"
  done
done

echo "=== T-1799 arm runs done ==="
