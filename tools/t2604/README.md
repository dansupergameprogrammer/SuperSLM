# T-2604 block-0 localization probe

`analyze.py` runs the same 14 inputs through three arms: Qwen3 float32, the
persisted exact Python integer model, and the C++ `RunLayerLoop` block-0 path.
It aborts on any C++/Python record or clamp-counter mismatch and writes the
float comparison to `result.json`.

Build the probe from a regular PowerShell prompt:

```bat
tools\t2604\build_probe.bat
```

Reproduce the committed result:

```powershell
$env:HF_HOME = 'D:\hf_cache'
$env:PYTHONPATH = 'tools'
python tools/t2604/analyze.py `
  --corpus D:\SuperEmbedder\.worktrees\te234-query-localization\tests\fixtures\c12b\corpus-239.jsonl `
  --integer-cache D:\SuperSLM\.worktrees\ask5-trackb\out\t2572_recalibrated `
  --hf-model D:\hf_cache\hub\models--Qwen--Qwen3-Embedding-0.6B\snapshots\97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 `
  --device cuda `
  --cpp-exe out\t2604-build\sslm_t2604_trace.exe `
  --sslm-model D:\SuperSLM\.worktrees\ask5-trackb\out\t2572_qwen3-embedding-0.6b.sslm `
  --cpp-dir out\t2604-cpp `
  --result tools\t2604\result.json
```

The JSONL dumps under `out/t2604-cpp/` are intentionally untracked. The
committed result contains their exact-record counts and the aggregate readings.
