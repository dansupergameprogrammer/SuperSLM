#!/bin/sh
# T-1836 -- decode-interior captures on T-1820's own six prompts, so the prefill half of every
# figure is a replication of an already-filed cell rather than a new population.
#
# Three stages, in this order because each needs the previous one's output:
#   1. engine, FREE-RUNNING (CPU). RunGreedyDecodeLoop's own trajectory, gated against
#      RunGreedyDecodeLoop itself. This is what the engine actually does.
#   2. float reference (GPU). Its own greedy continuation, gated against the model's own
#      forward at the first token.
#   3. engine, TEACHER-FORCED onto the float's continuation (CPU). Matched inputs on both
#      sides, which is what makes an interior state comparison a comparison at all.
#
# Prompt strings are $'...' literals passed as argv, never through $(...) -- command
# substitution strips the chat template's trailing newline (T-1788's tokenization-parity gate).
set -e
cd "$(dirname "$0")/.."

MODEL="D:/hf_cache/superslm_artifacts/qwen2.5-1.5b-instruct.sslm"
TOK="tests/fixtures/qwen2.5-1.5b.tok.sslm"
SYS="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MAXNEW="${MAXNEW:-32}"
STAGE="${1:-all}"

IDS="p1 p2 p3 q1 q2 q3"
prompt_for() {
  case "$1" in
    p1) echo "What is a good pillow choice for someone with back pain?" ;;
    p2) echo "How does a compass work?" ;;
    p3) echo "Write a short thank-you note after a job interview." ;;
    q1) echo "How do I keep houseplants alive in a low-light apartment?" ;;
    q2) echo "What is the difference between a virus and a bacterium?" ;;
    q3) echo "Give me a packing list for a weekend cycling trip in cold weather, and explain why each item matters." ;;
  esac
}

mkdir -p out/t1836

if [ "$STAGE" = "all" ] || [ "$STAGE" = "engine-free" ]; then
  for ID in $IDS; do
    U="$(prompt_for "$ID")"
    P=$'<|im_start|>system\n'"$SYS"$'<|im_end|>\n<|im_start|>user\n'"$U"$'<|im_end|>\n<|im_start|>assistant\n'
    echo "=== engine free-running $ID ==="
    ./out/t1836_decode_residual_probe.exe "$MODEL" "$TOK" "$P" "$ID" --dump-dir out/t1836 \
      --max-new "$MAXNEW" --suffix _free > "out/t1836/${ID}_free_engine.log" 2>&1
    tail -4 "out/t1836/${ID}_free_engine.log"
  done
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "float" ]; then
  for ID in $IDS; do
    U="$(prompt_for "$ID")"
    echo "=== float reference $ID ==="
    python -u tools/t1836_float_decode_reference.py "$U" --system "$SYS" --max-new "$MAXNEW" \
      --dump "out/t1836/${ID}_float.npy" > "out/t1836/${ID}_float.log" 2>&1
    tail -4 "out/t1836/${ID}_float.log"
  done
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "engine-forced" ]; then
  for ID in $IDS; do
    U="$(prompt_for "$ID")"
    P=$'<|im_start|>system\n'"$SYS"$'<|im_end|>\n<|im_start|>user\n'"$U"$'<|im_end|>\n<|im_start|>assistant\n'
    GEN="$(python -c "import json,sys;m=json.load(open(sys.argv[1]));print(','.join(str(t) for t in m['generated']))" "out/t1836/${ID}_float.npy.meta.json")"
    echo "=== engine teacher-forced $ID ==="
    ./out/t1836_decode_residual_probe.exe "$MODEL" "$TOK" "$P" "$ID" --dump-dir out/t1836 \
      --max-new "$MAXNEW" --suffix _forced --force-tokens "$GEN" \
      > "out/t1836/${ID}_forced_engine.log" 2>&1
    tail -5 "out/t1836/${ID}_forced_engine.log"
  done
fi

echo "=== t1836 stage '$STAGE' done ==="
