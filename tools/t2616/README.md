# T-2616 deep-layer localization probe

`analyze.py` captures the fourteen T-2604 float32 walks once, prepares exact
per-row carried-scale inputs, drives `sslm_t2616_probe`, and produces the
28-layer local/propagated budget, site walks, and one-site counterfactuals.
`cpp_layer_probe.cpp` runs either the normal full-stack path or one selected
layer from supplied hidden codes/scales while retaining that layer's causal K/V
history.

Build:

```bat
tools\t2616\build_probe.bat
```

The committed deciding outputs are:

- `budget.json`
- `sites-1.json`, `sites-27.json`
- `propagated-sites-14.json`, `propagated-sites-27.json`
- `attention-diagnosis-1.json`
- `counterfactual-local-1.json`
- `counterfactual-propagated-14.json`
- `counterfactual-local-27.json`

The generated float capture, local-input slices, and C++ JSONL traces live
under `out/t2616/capture/` and are intentionally untracked. All commands are
available from `python tools/t2616/analyze.py --help`; the derivation record
pins the exact population, artifact, and interpretation.
