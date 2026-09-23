"""Apply one T-2851 production mutant in the isolated scratch worktree.

Use from D:/_slm170 with --tree D:/_t2956-mutants/src. Every invocation
restores the prior production mutation first; this script never touches the
main checkout. The caller builds and executes the named test cell afterward.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

FILES = (
    "src/forward/forward_sites.cpp",
    "src/sslm_abi.cpp",
    "src/gpu/gpu_1p0.cpp",
    "src/gpu/superslm_gpu.cpp",
    "src/gpu/d3d12_harness.h",
    "src/gpu/shaders/logits_site.hlsl",
)


class Patch:
    def __init__(self, tree: Path):
        self.tree = tree
        self.changed: set[str] = set()

    def replace(self, name: str, old: str, new: str):
        path = self.tree / name
        content = path.read_text(encoding="utf-8")
        count = content.count(old)
        if count != 1:
            raise RuntimeError(f"{name}: expected one landing, found {count}: {old[:80]!r}")
        path.write_text(content.replace(old, new, 1), encoding="utf-8", newline="\n")
        self.changed.add(name)


def apply(p: Patch, ident: str):
    f = "src/forward/forward_sites.cpp"
    g = "src/gpu/gpu_1p0.cpp"
    s = "src/gpu/superslm_gpu.cpp"
    h = "src/gpu/d3d12_harness.h"
    shader = "src/gpu/shaders/logits_site.hlsl"
    if ident == "a":
        p.replace(f, "const size_t task_count = (vocab_size + rows_per_task - 1) / rows_per_task;",
                  "const size_t task_count = vocab_size / rows_per_task;")
    elif ident == "c":
        p.replace(f, "if (logits[i] > best_value) {",
                  "if (logits[i] >= best_value) { // MUTANT c: last-wins tie merge")
    elif ident == "c_exact":
        # Move the winner into each parallel task, merge local winners with >=
        # (later partition wins ties), and consume that merged result at the
        # existing serial caller. The TLS channel preserves the public ABI in
        # this disposable scratch build; it is keyed to the exact row pointer.
        p.replace(f, "namespace {\n\n// Call-local state one LogitsSiteParallel",
                  "namespace {\n\nthread_local const int32_t* g_task_argmax_row = nullptr; // MUTANT c_exact\n"
                  "thread_local int32_t g_task_argmax = 0;\n\n"
                  "// Call-local state one LogitsSiteParallel")
        p.replace(f, "\tstd::atomic<bool>* violation;\n};\n\nvoid LogitsTask",
                  "\tstd::atomic<bool>* violation;\n"
                  "\tint64_t* local_values;\n\tint32_t* local_indices;\n};\n\nvoid LogitsTask")
        p.replace(f, "\t                      end - begin, c.wide_logits + begin);\n\t// Release:",
                  "\t                      end - begin, c.wide_logits + begin);\n"
                  "\tint64_t local_best = c.wide_logits[begin];\n"
                  "\tint32_t local_index = static_cast<int32_t>(begin);\n"
                  "\tfor (size_t row = begin + 1; row < end; ++row) {\n"
                  "\t\tif (c.wide_logits[row] > local_best) {\n"
                  "\t\t\tlocal_best = c.wide_logits[row];\n"
                  "\t\t\tlocal_index = static_cast<int32_t>(row);\n"
                  "\t\t}\n\t}\n"
                  "\tc.local_values[task_index] = local_best;\n"
                  "\tc.local_indices[task_index] = local_index;\n"
                  "\t// Release:")
        p.replace(f, "\tif (pf == nullptr || pf->run == nullptr || pf->max_tasks <= 1 || vocab_size == 0) {",
                  "\tg_task_argmax_row = nullptr;\n"
                  "\tif (pf == nullptr || pf->run == nullptr || pf->max_tasks <= 1 || vocab_size == 0) {")
        p.replace(f, "\tstd::atomic<bool> violation{false};\n\tLogitsTaskCtx ctx{",
                  "\tstd::atomic<bool> violation{false};\n"
                  "\tint64_t local_values[SSLM_PARALLEL_FOR_MAX_TASKS]{};\n"
                  "\tint32_t local_indices[SSLM_PARALLEL_FOR_MAX_TASKS]{};\n"
                  "\tLogitsTaskCtx ctx{")
        p.replace(f, "static_cast<int32_t>(task_count), state, &violation};",
                  "static_cast<int32_t>(task_count), state, &violation, local_values, local_indices};")
        p.replace(f, "\tif (!complete) return SslmForwardStatus::ParallelForIncomplete;\n"
                  "\treturn NarrowRowChecked(wide_logits, vocab_size, out_logits);",
                  "\tif (!complete) return SslmForwardStatus::ParallelForIncomplete;\n"
                  "\tconst auto narrow = NarrowRowChecked(wide_logits, vocab_size, out_logits);\n"
                  "\tif (narrow != SslmForwardStatus::Ok) return narrow;\n"
                  "\tint64_t best = INT64_MIN;\n"
                  "\tfor (size_t task = 0; task < task_count; ++task) {\n"
                  "\t\tif (local_values[task] >= best) { // MUTANT c_exact: last partition wins ties\n"
                  "\t\t\tbest = local_values[task];\n"
                  "\t\t\tg_task_argmax = local_indices[task];\n"
                  "\t\t}\n\t}\n"
                  "\tg_task_argmax_row = out_logits;\n"
                  "\treturn SslmForwardStatus::Ok;")
        p.replace(f, "int32_t ArgmaxLowestIndexTieBreak(const int32_t* logits, size_t n) {\n"
                  "\tint32_t best_index = 0;",
                  "int32_t ArgmaxLowestIndexTieBreak(const int32_t* logits, size_t n) {\n"
                  "\tif (logits == g_task_argmax_row) {\n"
                  "\t\tg_task_argmax_row = nullptr;\n"
                  "\t\treturn g_task_argmax; // MUTANT c_exact: task-local winner consumed\n"
                  "\t}\n"
                  "\tint32_t best_index = 0;")
    elif ident == "q_cpu":
        p.replace("src/sslm_abi.cpp",
                  "\t\t\tMaybeInjectCpuFinishDegenerateLogitRow(logit_row, static_cast<int32_t>(c.vocab_size));",
                  "\t\t\t// MUTANT q_cpu: armed seam never reaches the real finish row.")
    elif ident == "q_gpu":
        p.replace(g, "\tMaybeInjectGpuFinishDegenerateLogitRow(logit_row.data(), model->vocab_size);",
                  "\t// MUTANT q_gpu: armed seam never reaches the real finish row.")
    elif ident == "b":
        p.replace(f, "if (!complete) return SslmForwardStatus::ParallelForIncomplete;",
                  "(void)complete; // MUTANT b: omit exactly-once refusal")
    elif ident == "b2":
        p.replace(f, "std::atomic<bool>* violation;", "std::atomic<bool>* violation;\n\tstd::atomic<int32_t>* calls;")
        p.replace(f, "\tuint8_t expected = 0;\n\tif (!c.state[task_index].compare_exchange_strong(expected, uint8_t{1},\n\t                                                 std::memory_order_acq_rel)) {\n\t\tc.violation->store(true, std::memory_order_release);\n\t\treturn;\n\t}",
                  "\tc.calls->fetch_add(1, std::memory_order_relaxed);\n\tc.state[task_index].store(uint8_t{1}, std::memory_order_relaxed);")
        p.replace(f, "std::atomic<bool> violation{false};", "std::atomic<bool> violation{false};\n\tstd::atomic<int32_t> calls{0};")
        p.replace(f, "static_cast<int32_t>(task_count), state, &violation};",
                  "static_cast<int32_t>(task_count), state, &violation, &calls};")
        p.replace(f, "\tfor (size_t i = 0; complete && i < task_count; ++i) {\n\t\tcomplete = state[i].load(std::memory_order_acquire) == 2;\n\t}",
                  "\tcomplete = complete && calls.load(std::memory_order_acquire) == static_cast<int32_t>(task_count);")
    elif ident == "d":
        p.replace(shader, "for (uint s = LANES >> 1; s > 0u; s >>= 1)",
                  "for (uint s = LANES >> 2; s > 0u; s >>= 1)")
    elif ident == "e":
        p.replace(g, "const bool head_on_device = (cfg.flags & SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE) != 0;",
                  "const bool head_on_device = false; // MUTANT e: ignore flag")
    elif ident == "f":
        p.replace(g, "if ((cfg.flags & ~SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE) != 0) {",
                  "if (false && (cfg.flags & ~SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE) != 0) {")
    elif ident == "g":
        p.replace("src/sslm_abi.cpp", "\treturn st;\n}\n\n// G5-2/G5-5 (design Sec4/Sec6, T-2132):",
                  "\tif (ws && ws->parallel_for.run && *consumed > 0)\n"
                  "\t\tws->parallel_for.run(ws->parallel_for.host_ctx, 1,\n"
                  "\t\t    +[](void*, int32_t) {}, nullptr); // MUTANT g\n"
                  "\treturn st;\n}\n\n// G5-2/G5-5 (design Sec4/Sec6, T-2132):")
    elif ident == "h":
        p.replace(g, "ctx, reinterpret_cast<const int8_t*>(head_w->data), static_cast<uint32_t>(vocab_size),",
                  "ctx, reinterpret_cast<const int8_t*>(embed_w->data), static_cast<uint32_t>(vocab_size),")
    elif ident == "h2":
        p.replace(g, "h->head_rows = h->head_weights.data();",
                  "h->head_rows = h->embed_weights.data(); // MUTANT h2")
    elif ident == "i":
        p.replace(g, "fst = superslm::NarrowRowChecked(wide_logits.data(), static_cast<size_t>(model->vocab_size),\n\t\t                                 logit_row.data());",
                  "for (size_t i = 0; i < logit_row.size(); ++i)\n\t\t\tlogit_row[i] = static_cast<int32_t>(wide_logits[i]);\n\t\tfst = superslm::SslmForwardStatus::Ok;")
    elif ident == "j":
        p.replace(g, "if (hr == E_OUTOFMEMORY && !MapDeviceReportedRemoved(dev)) {\n\t\t\t\t\treturn SSLM_GPU_ALLOCATION_FAILED;",
                  "if (hr == E_OUTOFMEMORY && !MapDeviceReportedRemoved(dev)) {\n\t\t\t\t\treturn SSLM_DEVICE_LOST; // MUTANT j")
    elif ident == "k":
        p.replace(g, "\t\t\t\treturn SSLM_DEVICE_LOST;\n\t\t\t}\n\t\t}",
                  "\t\t\t\treturn SSLM_GPU_ALLOCATION_FAILED; // MUTANT k\n\t\t\t}\n\t\t}")
    elif ident.startswith("l") and len(ident) == 2 and ident[1] in "123456":
        slot = int(ident[1])
        p.replace(g, "\t\tfor (const Allocation& a : allocations) {\n\t\t\tconst HRESULT hr = dev.TryMakeBuffer(a.bytes, a.heap, a.flags, a.state, a.target);",
                  "\t\tuint32_t slot = 0;\n\t\tfor (const Allocation& a : allocations) {\n"
                  f"\t\t\tif (++slot == {slot}) {{\n"
                  "\t\t\t\t*a.target = dev.MakeBuffer(a.bytes, a.heap, a.flags, a.state);\n"
                  "\t\t\t\tcontinue; // MUTANT l: throwing allocation\n\t\t\t}\n"
                  "\t\t\tconst HRESULT hr = dev.TryMakeBuffer(a.bytes, a.heap, a.flags, a.state, a.target);")
    elif ident == "m":
        p.replace(g, "} else if (base->config.tie_word_embeddings) {\n\t\th->head_rows = h->embed_weights.data();",
                  "} else if (base->config.tie_word_embeddings) {\n\t\th->head_weights = h->embed_weights; // MUTANT m: duplicate tied copy\n\t\th->head_rows = h->head_weights.data();")
    elif ident == "n":
        p.replace(g, "\t\tfor (const Allocation& a : allocations) {",
                  "\t\tSSLM_GPU_HR(dev.alloc->Reset());\n"
                  "\t\tSSLM_GPU_HR(dev.list->Reset(dev.alloc.Get(), nullptr));\n"
                  "\t\tfor (const Allocation& a : allocations) { // MUTANT n")
        p.replace(g, "\t\t// Phase B.\n\t\tSSLM_GPU_HR(dev.alloc->Reset());\n\t\tSSLM_GPU_HR(dev.list->Reset(dev.alloc.Get(), nullptr));",
                  "\t\t// Phase B reset already occurred before allocations (MUTANT n).")
    elif ident in ("n2_upload", "n3_upload"):
        old = ("\tMicrosoft::WRL::ComPtr<ID3D12Resource> upload = dev.Upload(data, bytes);\n"
               "\tMicrosoft::WRL::ComPtr<ID3D12Resource> resident =\n"
               "\t    dev.MakeBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT, resource_flags, D3D12_RESOURCE_STATE_COPY_DEST);\n"
               "\t// Phase B: reset, record, Close (with one retry), execute, wait.\n"
               "\tSSLM_GPU_HR(dev.alloc->Reset());\n\tSSLM_GPU_HR(dev.list->Reset(dev.alloc.Get(), nullptr));")
        resident = ("\tMicrosoft::WRL::ComPtr<ID3D12Resource> resident =\n"
                    "\t    dev.MakeBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT, resource_flags, D3D12_RESOURCE_STATE_COPY_DEST);\n")
        upload = "\tMicrosoft::WRL::ComPtr<ID3D12Resource> upload = dev.Upload(data, bytes);\n"
        reset = "\tSSLM_GPU_HR(dev.alloc->Reset());\n\tSSLM_GPU_HR(dev.list->Reset(dev.alloc.Get(), nullptr));\n"
        new = (reset + upload + resident if ident == "n2_upload"
               else resident + reset + upload)
        p.replace(g, old, new + f"\t// MUTANT {ident}")
    elif ident in ("n2_restore", "n3_restore"):
        old = ("\t\tauto upload_buf = dev.Upload(src, workspace_size);\n"
               "\t\tauto device_buf = dev.MakeBuffer(workspace_size, D3D12_HEAP_TYPE_DEFAULT,\n"
               "\t\t                                  D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);\n"
               "\t\tauto readback_buf = dev.MakeBuffer(workspace_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,\n"
               "\t\t                                    D3D12_RESOURCE_STATE_COPY_DEST);\n"
               "\t\tSSLM_GPU_HR(dev.alloc->Reset());\n\t\tSSLM_GPU_HR(dev.list->Reset(dev.alloc.Get(), nullptr));")
        device = ("\t\tauto device_buf = dev.MakeBuffer(workspace_size, D3D12_HEAP_TYPE_DEFAULT,\n"
                  "\t\t                                  D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);\n")
        upload = "\t\tauto upload_buf = dev.Upload(src, workspace_size);\n"
        readback = ("\t\tauto readback_buf = dev.MakeBuffer(workspace_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,\n"
                    "\t\t                                    D3D12_RESOURCE_STATE_COPY_DEST);\n")
        reset = "\t\tSSLM_GPU_HR(dev.alloc->Reset());\n\t\tSSLM_GPU_HR(dev.list->Reset(dev.alloc.Get(), nullptr));\n"
        new = (reset + upload + device + readback if ident == "n2_restore"
               else device + reset + upload + readback)
        p.replace(s, old, new + f"\t\t// MUTANT {ident}")
    elif ident == "o":
        p.replace(g, "\t\tfor (uint32_t k = 0; k < b.H; ++k) b.x_upload_ptr[k] = x_codes[k];",
                  "\t\tMicrosoft::WRL::ComPtr<ID3D12Resource> fresh;\n\t\tSSLM_GPU_HR(dev.TryMakeBuffer(static_cast<UINT64>(b.H) * 4, D3D12_HEAP_TYPE_UPLOAD,\n\t\t    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &fresh));\n\t\tfor (uint32_t k = 0; k < b.H; ++k) b.x_upload_ptr[k] = x_codes[k];")
    elif ident == "o2":
        p.replace(g, "struct SslmGpuDeviceLogitsBundleForTest {\n\tDeviceLogitsBuffers buffers;\n};",
                  "struct SslmGpuDeviceLogitsBundleForTest {\n\tDeviceLogitsBuffers buffers;\n"
                  "\tstd::vector<int8_t> head_copy; // MUTANT o2\n};")
        p.replace(g, "auto bundle = std::make_unique<SslmGpuDeviceLogitsBundleForTest>();\n\t\tconst SslmGpuStatus st = CreateDeviceLogitsBuffers(ctx, head_rows, V, H, &bundle->buffers);",
                  "auto bundle = std::make_unique<SslmGpuDeviceLogitsBundleForTest>();\n"
                  "\t\tbundle->head_copy.assign(head_rows, head_rows + static_cast<size_t>(V) * H);\n"
                  "\t\tconst SslmGpuStatus st = CreateDeviceLogitsBuffers(ctx, head_rows, V, H, &bundle->buffers);")
        p.replace(g, "\treturn InvokeGpuApiBoundary(\n\t    __func__, [&] { return RunDeviceLogits(ctx, bundle->buffers, x_codes, wide_out); });",
                  "\treturn InvokeGpuApiBoundary(__func__, [&] {\n"
                  "\t\tDeviceLogitsBuffers fresh;\n"
                  "\t\tconst SslmGpuStatus made = CreateDeviceLogitsBuffers(ctx, bundle->head_copy.data(),\n"
                  "\t\t    bundle->buffers.V, bundle->buffers.H, &fresh);\n"
                  "\t\tif (made != SSLM_OK) return made;\n"
                  "\t\treturn RunDeviceLogits(ctx, fresh, x_codes, wide_out); // MUTANT o2\n"
                  "\t});")
    elif ident == "p":
        p.replace(h, "\t\tconst HRESULT retry = CloseListOnce();\n\t\tstd::fprintf(stderr, \"superslm_gpu: command list Close failed; retry %s\\n\",\n\t\t             SUCCEEDED(retry) ? \"succeeded (list Closed)\" : \"failed (list left recording)\");",
                  "\t\t// MUTANT p: no second Close attempt; list stays recording.")
    else:
        raise ValueError(f"unknown mutant {ident}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ident", help="mutant id, or reset")
    parser.add_argument("--tree", type=Path, default=Path(r"D:\_t2956-mutants\src"))
    args = parser.parse_args()
    tree = args.tree.resolve()
    if tree != Path(r"D:\_t2956-mutants\src").resolve():
        raise RuntimeError(f"refusing to edit a tree outside the dedicated scratch worktree: {tree}")
    if subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True,
                      text=True, check=True).stdout.strip() != "c9081e0fd33fccabf52e3fb247ad2f51efb74e6e":
        raise RuntimeError("scratch worktree is not at the pinned builder commit")
    subprocess.run(["git", "-C", str(tree), "restore", "--", *FILES], check=True)
    if args.ident == "reset":
        print("RESET")
        return
    patch = Patch(tree)
    apply(patch, args.ident)
    for name in sorted(patch.changed):
        digest = hashlib.sha256((tree / name).read_bytes()).hexdigest()
        print(f"MUTANT {args.ident} {name} sha256={digest}")


if __name__ == "__main__":
    main()
