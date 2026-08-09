"""Stage 4 -- Q4: measured throughput, in the same units as the campaign's own
GPU cost table (seconds per document, one forward per document, batch width 1).

`Claude/Vitruvius/t1870-encoder-retrieval-instrument-design-2026-08-09.md` S12.4
cites T-1777's own plain single-item forward rate as its Stage-2 lower bound:
3.30 s/doc bf16, 3.02 s/doc fp32 (averaged to 3.16 s/doc). This stage runs the
identical shape of measurement -- one forward pass per document, batch width 1,
no batching across documents -- on THIS device, at both dtypes, over the bundled
real corpus subset, so its output numbers drop directly into that table's own
columns without unit conversion.

The first document at each dtype is timed separately and excluded from the
mean: it carries one-time cost (HIP kernel selection/compilation, memory pool
warm-up) that the campaign's own harness also does not count against its
steady-state rate.
"""
import statistics
import sys
import time
import traceback

from common import emit, fail, ok, load_prompts, model_dir

STAGE = "throughput"
N_DOCS = 12


def run_pass(model, tokenizer, prompts, dtype_name):
    times = []
    import torch
    for i, rec in enumerate(prompts):
        text = rec["text"]
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)
        pooled_norm = float(out.hidden_states[-1][0].mean(dim=0).norm().item())
        print(f"[{STAGE}][{dtype_name}] doc {i+1}/{len(prompts)} "
              f"({'warmup, excluded' if i == 0 else 'timed'}): "
              f"{dt:.4f}s  pooled_norm={pooled_norm:.4f}")
    warmup, timed = times[0], times[1:]
    return warmup, timed


def main() -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        emit(fail(STAGE, "torch or transformers did not import",
                  {"exception": repr(e)}))
        return 1

    if not torch.cuda.is_available():
        emit(fail(STAGE, "no device available -- see stage_import"))
        return 1

    mdir = model_dir()
    prompts = load_prompts(limit=N_DOCS)
    if len(prompts) < 2:
        emit(fail(STAGE, "need at least 2 bundled prompts (1 warmup + 1 timed)"))
        return 1

    detail = {"n_docs": len(prompts), "model_dir": str(mdir)}
    results_by_dtype = {}

    for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        print(f"[{STAGE}] === loading model in {dtype_name} ===")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(mdir), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(mdir), local_files_only=True, torch_dtype=dtype
            )
            model.to("cuda:0")
            model.eval()
        except Exception as e:
            print(f"[{STAGE}] model load ({dtype_name}) raised: {e!r}")
            traceback.print_exc()
            detail[f"{dtype_name}_load_exception"] = repr(e)
            results_by_dtype[dtype_name] = {"error": repr(e)}
            continue

        try:
            warmup, timed = run_pass(model, tokenizer, prompts, dtype_name)
            mean_s = statistics.mean(timed)
            median_s = statistics.median(timed)
            stdev_s = statistics.stdev(timed) if len(timed) > 1 else 0.0
            results_by_dtype[dtype_name] = {
                "warmup_seconds": warmup,
                "timed_seconds_per_doc": timed,
                "mean_s_per_doc": mean_s,
                "median_s_per_doc": median_s,
                "stdev_s_per_doc": stdev_s,
                "n_timed": len(timed),
            }
            print(f"[{STAGE}] {dtype_name}: mean={mean_s:.4f} s/doc "
                  f"median={median_s:.4f} s/doc stdev={stdev_s:.4f} "
                  f"over {len(timed)} timed documents (warmup {warmup:.4f}s excluded)")
        except Exception as e:
            print(f"[{STAGE}] timing loop ({dtype_name}) raised: {e!r}")
            traceback.print_exc()
            results_by_dtype[dtype_name] = {"error": repr(e), "traceback": traceback.format_exc()}
        finally:
            del model
            torch.cuda.empty_cache()

    detail["by_dtype"] = results_by_dtype
    detail["design_cost_table_reference"] = (
        "Claude/Vitruvius/t1870-encoder-retrieval-instrument-design-2026-08-09.md "
        "S12.4: lower-bound rate on RTX 2080 Super (T-1777) = 3.30 s/doc bf16, "
        "3.02 s/doc fp32. Compare this stage's mean_s_per_doc figures directly "
        "against those two numbers -- same shape of measurement, same units."
    )

    any_ok = any("mean_s_per_doc" in v for v in results_by_dtype.values())
    if any_ok:
        emit(ok(STAGE, detail))
        return 0
    else:
        emit(fail(STAGE, "throughput measurement failed at both dtypes", detail))
        return 1


if __name__ == "__main__":
    sys.exit(main())
