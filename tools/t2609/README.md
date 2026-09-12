# T-2609 score-to-softmax localization probe

`analyze.py` reuses the T-2604 block-0 walk on the calibration-fixed artifact,
executes exact-softmax and float-softmax substitutions, and reads selected
attention sites from the real C++ full-stack path at layers 1, 14, and 27.

Build the filtered full-stack probe:

```bat
tools\t2609\build_probe.bat
```

Reproduce the result from the engine worktree:

```powershell
$env:HF_HOME = 'D:\hf_cache'
$env:PYTHONPATH = 'tools'
python tools/t2609/analyze.py `
  --corpus D:\SuperEmbedder\.worktrees\te234-query-localization\tests\fixtures\c12b\corpus-239.jsonl `
  --integer-cache D:\SuperSLM\.worktrees\t2607-k-calibration-fix\out\t2607_recalibrated `
  --hf-model D:\hf_cache\hub\models--Qwen--Qwen3-Embedding-0.6B\snapshots\97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 `
  --sslm-model D:\SuperSLM\.worktrees\t2607-k-calibration-fix\out\t2607_qwen3-embedding-0.6b.sslm `
  --cpp-exe out\t2609\sslm_t2609_trace.exe `
  --cpp-dir out\t2609\cpp `
  --fixed-result D:\SuperSLM\.worktrees\t2607-k-calibration-fix\out\t2604-result.json `
  --result tools\t2609\result.json
```

The JSONL dumps under `out/t2609/cpp/` are intentionally untracked.
