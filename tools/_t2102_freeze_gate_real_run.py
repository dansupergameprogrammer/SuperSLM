"""T-2102 (D-SLM3304): runs the SHIPPED, corrected freeze-health gate (run_freeze_health_gate,
tools/sslm_convert_adapter.py, D-SLM3295/D-SLM3296 -- T-2099's joint-power repair, closed at
c8dbffb) against the REAL shopkeeper LoRA v2 adapter (D:\\hf_cache\\superslm_artifacts\\
qwen2.5-1.5b-shopkeeper-lora-v2\\final), every one of its adapted (layer, proj) pairs (28 layers
x 7 target_modules = 196 pairs -- adapter_config.json's own target_modules is all seven).

Not weakened, bypassed, or silently overridden: this calls run_freeze_health_gate exactly as
shipped, at its own default pilot_n=200, adapter-level budget=1800s, escalation factor=2,
pilot_n_ceiling=60000 -- no argument override. `collect_pair(name, pilot_n)` is one real call to
_b3_collect_pair_raw_draws per pair (the SAME production collector T-2099's own probe used),
returning (composed_pilot, composed_val) as (candidate, reference) -- disjoint seed_base/
validation_seed_base streams of the SAME real pair, per D-SLM3295's own "independence is a
property of the collection."

Read-only against D:\\SuperSLM except this file's own output. Writes t2102_freeze_verdict.json
next to this script as soon as the gate returns (whatever the verdict).

CHECKPOINTED (added after this run's own first attempt was killed mid-run by the harness's
background-task lifetime limit at 140/196 pairs, ~54 minutes in -- an external process-management
event, not a script defect; the SAME resumability shape T-2046/T-2065 already built for the
identical real-196-pair-sweep cost, tools/sslm_convert_adapter.py's own `checkpoint_path` parameter
docstring). Every pair's own raw draws are appended to a JSONL checkpoint keyed by (name, pilot_n)
the instant they are computed; on startup, any pair already checkpointed at the CURRENT pilot_n is
read back instead of recomputed. A restart after a second kill re-pays only the pairs not yet
checkpointed at whatever pilot_n step is in progress.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

BUILD_TOOLS = Path(r"D:\SuperSLM\.worktrees\t2021-build\tools")
SPIKE_ROOT = Path(r"D:\Wizard\Tools")
ADAPTER_DIR = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-shopkeeper-lora-v2\final")
BASE_SSLM = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm")

sys.path.insert(0, str(BUILD_TOOLS))
sys.path.insert(0, str(SPIKE_ROOT))
import sslm_convert_adapter as A  # noqa: E402
from superslm_spike import pipeline  # noqa: E402

OUT_PATH = Path(__file__).parent / "t2102_freeze_verdict.json"
PROGRESS_PATH = Path(__file__).parent / "t2102_freeze_progress.txt"
CHECKPOINT_PATH = Path(__file__).parent / "_t2102_freeze_checkpoint.jsonl"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main():
    t_start = time.time()
    log("T-2102 real freeze-health-gate run starting")
    log(f"adapter={ADAPTER_DIR}")
    log(f"base={BASE_SSLM}")

    meta = A.load_adapter_meta(ADAPTER_DIR)
    log(f"adapter meta: rank={meta.rank} lora_alpha={meta.lora_alpha} use_rslora={meta.use_rslora} "
        f"target_modules={meta.target_modules}")

    checkpoint_dir = A._resolve_base_checkpoint_dir(ADAPTER_DIR)
    log(f"raw checkpoint dir (float reference): {checkpoint_dir}")
    cfg = pipeline.load_config(checkpoint_dir / "config.json")
    tensors = pipeline._open_checkpoint_tensors(checkpoint_dir)
    log(f"base geometry: num_hidden_layers={cfg.num_hidden_layers}")

    pair_names = []
    for layer in range(cfg.num_hidden_layers):
        for proj in A.PEFT_ADAPTABLE_PROJECTIONS:
            if proj in meta.target_modules:
                pair_names.append(f"layer{layer}.{proj}")
    log(f"{len(pair_names)} adapted pairs (28 layers x {len(meta.target_modules)} target_modules)")

    pair_cache = {}

    def _pair_inputs(name):
        if name not in pair_cache:
            layer_str, proj = name.split(".", 1)
            layer = int(layer_str[len("layer"):])
            w_f = A.read_base_projection_weight(tensors, layer, proj)
            a_f, b_scaled = A.read_peft_lora_pair(ADAPTER_DIR, layer, proj, meta)
            Wc, w_scales = pipeline.quantize_weight_per_channel(w_f, output_axis=0)
            w = np.asarray(w_scales, dtype=np.float64)
            S = float(np.max(w))
            Ac, alpha_scales = pipeline.quantize_weight_per_channel(a_f, output_axis=0)
            alpha = np.asarray(alpha_scales, dtype=np.float64)
            Bc, beta_scales = pipeline.quantize_weight_per_channel(b_scaled, output_axis=0)
            beta = np.asarray(beta_scales, dtype=np.float64)
            g = np.random.default_rng(0xC0FFEE)
            xf = g.standard_normal(w_f.shape[1])
            xmax = float(np.max(np.abs(xf)))
            X = xmax / 127.0 if xmax > 0.0 else 1.0
            xc = np.clip(np.round(xf / X), -127, 127).astype(np.int64)
            T = A.compute_t(list(alpha), list(np.abs(Ac.astype(np.int64) @ xc)))
            pair_cache[name] = (w_f, a_f, b_scaled, Wc, w, S, Ac, alpha, Bc, beta, T)
        return pair_cache[name]

    # --- checkpoint: load whatever was already computed by a prior (killed) attempt. ------------
    checkpoint = {}  # (name, pilot_n) -> (composed_pilot, composed_val)
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                checkpoint[(rec["name"], rec["pilot_n"])] = (
                    np.asarray(rec["composed_pilot"], dtype=np.float64),
                    np.asarray(rec["composed_val"], dtype=np.float64))
        log(f"resuming: {len(checkpoint)} (pair, pilot_n) draws already checkpointed at {CHECKPOINT_PATH}")
    checkpoint_file = open(CHECKPOINT_PATH, "a", encoding="utf-8")

    pairs_done = [0]

    def collect_pair(name, pilot_n):
        key = (name, pilot_n)
        if key in checkpoint:
            composed_pilot, composed_val = checkpoint[key]
        else:
            w_f, a_f, b_scaled, Wc, w, S, Ac, alpha, Bc, beta, T = _pair_inputs(name)
            raw = A._b3_collect_pair_raw_draws(w_f, a_f, b_scaled, Wc, w, S, Ac, alpha, Bc, beta, T,
                                               pilot_n=pilot_n)
            composed_pilot, composed_val = raw["composed_pilot"], raw["composed_val"]
            checkpoint[key] = (composed_pilot, composed_val)
            checkpoint_file.write(json.dumps({"name": name, "pilot_n": pilot_n,
                                              "composed_pilot": composed_pilot.tolist(),
                                              "composed_val": composed_val.tolist()}) + "\n")
            checkpoint_file.flush()
        pairs_done[0] += 1
        if pairs_done[0] % 20 == 0 or pairs_done[0] == len(pair_names):
            log(f"  collected {pairs_done[0]}/{len(pair_names)} pairs at pilot_n={pilot_n} "
                f"({time.time() - t_start:.0f}s elapsed)")
        return composed_pilot, composed_val

    def projection_type_of(name):
        return name.split(".", 1)[1]

    log("invoking run_freeze_health_gate -- SHIPPED defaults, no override "
        f"(pilot_n={A._B3_PILOT_N}, budget_seconds={A._FREEZE_ESCALATION_BUDGET_SECONDS}, "
        f"pilot_n_ceiling={A._FREEZE_PILOT_N_CEILING}, escalation_factor={A._FREEZE_ESCALATION_FACTOR})")

    result = A.run_freeze_health_gate(collect_pair, pair_names, projection_type_of=projection_type_of,
                                      verbose=True)

    total_elapsed = time.time() - t_start
    log(f"DONE: overall={result['overall']} freeze_allowed={result['freeze_allowed']} "
        f"stop_reason={result['stop_reason']} escalation_steps={result['escalation_steps']} "
        f"final_pilot_n={result['final_pilot_n']} seconds_spent={result['seconds_spent']:.1f} "
        f"total_wallclock={total_elapsed:.1f}s")

    out = {
        "adapter": str(ADAPTER_DIR), "base": str(BASE_SSLM), "n_pairs": len(pair_names),
        "meta": {"rank": meta.rank, "lora_alpha": meta.lora_alpha, "use_rslora": meta.use_rslora,
                 "target_modules": list(meta.target_modules)},
        "overall": result["overall"], "freeze_allowed": result["freeze_allowed"],
        "stop_reason": result["stop_reason"], "escalation_steps": result["escalation_steps"],
        "final_pilot_n": result["final_pilot_n"], "seconds_spent": result["seconds_spent"],
        "total_wallclock_seconds": total_elapsed,
        "tier1": result["tier1"], "tier2": result["tier2"],
    }
    OUT_PATH.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    log(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
