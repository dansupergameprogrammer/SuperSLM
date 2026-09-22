#include "superslm/sslm_abi.h"
#define SUPERSLM_T2791_SUPPRESS_GPU_STATUS_USING_ENUM
#include "../t2791-gpu-prefill-read-red-suite/fixture_common.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void Cell(const char* id, bool held, const char* reason) {
    ++g_checks;
    if (!held) ++g_failures;
    std::printf("CELL %s status=%s reason=%s\n", id, held ? "GREEN" : "RED", reason);
}

bool ReadFile(const char* path, std::vector<uint8_t>* bytes) {
    FILE* file = nullptr;
    if (fopen_s(&file, path, "rb") != 0 || !file) return false;
    std::fseek(file, 0, SEEK_END);
    const long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    bytes->resize(size > 0 ? static_cast<size_t>(size) : 0);
    const bool ok = size >= 0 &&
        (size == 0 || std::fread(bytes->data(), 1, bytes->size(), file) == bytes->size());
    std::fclose(file);
    return ok;
}

uint32_t Le32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

uint64_t Le64(const uint8_t* p) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= uint64_t(p[i]) << (8 * i);
    return value;
}

bool RetargetAdapter(std::vector<uint8_t>* bytes,
                     const std::array<uint8_t, superslm::kIntegrityHashBytes>& base_hash) {
    if (!bytes || bytes->size() < 64) return false;
    const uint32_t sections = Le32(bytes->data() + 12);
    size_t provenance = 0;
    for (uint32_t i = 0; i < sections; ++i) {
        const size_t row = 64 + size_t(i) * 40;
        if (row + 40 > bytes->size()) return false;
        if (Le32(bytes->data() + row) == 1) provenance = static_cast<size_t>(Le64(bytes->data() + row + 8));
    }
    if (!provenance || provenance + 48 > bytes->size()) return false;
    std::copy(base_hash.begin(), base_hash.end(), bytes->begin() + provenance + 16);
    std::fill(bytes->begin() + 32, bytes->begin() + 64, uint8_t{0});
    uint8_t digest[32];
    superslm::Sha256Hash(bytes->data(), bytes->size(), digest);
    std::copy(digest, digest + 32, bytes->begin() + 32);
    return true;
}

enum class SchemaOp { Bind, Rebind, Unbind };
const char* OpName(SchemaOp op) {
    return op == SchemaOp::Bind ? "bind" : op == SchemaOp::Rebind ? "rebind" : "unbind";
}

struct CpuEnv {
    std::vector<uint8_t> model_bytes;
    std::vector<uint8_t> adapter_bytes;
    superslm::SslmModelView view{};
    sslm_model model = nullptr;
    sslm_kv_pool pool = nullptr;
    sslm_workspace workspace = nullptr;
    sslm_schema schema_a = nullptr;
    sslm_schema schema_b = nullptr;
    sslm_adapter adapter = nullptr;
    std::vector<uint8_t> pool_storage;
    std::vector<uint8_t> workspace_storage;

    bool Open(const char* model_path, const char* adapter_path) {
        std::string error;
        if (!ReadFile(model_path, &model_bytes) ||
            superslm::SslmModel::Load(model_bytes.data(), model_bytes.size(), view, &error) !=
                superslm::SslmModelStatus::Ok ||
            sslm_model_map(model_bytes.data(), model_bytes.size(), &model) != SSLM_OK) {
            std::fprintf(stderr, "CPU model open failed: %s\n", error.c_str());
            return false;
        }
        constexpr int32_t kBlocks = 32;
        const size_t block = sslm_kv_block_size(model);
        const size_t pool_bytes = block * kBlocks + sslm_kv_pool_overhead_size(model, kBlocks);
        pool_storage.resize(pool_bytes + 63);
        void* pool_aligned = pool_storage.data();
        size_t pool_space = pool_storage.size();
        if (!std::align(64, pool_bytes, pool_aligned, pool_space) ||
            sslm_kv_pool_create(model, pool_aligned, pool_bytes, kBlocks, &pool) != SSLM_OK) {
            return false;
        }
        sslm_config config{};
        config.max_batch = 1;
        config.max_chunk_budget = 8;
        config.max_layer_budget = static_cast<int32_t>(view.config.num_hidden_layers);
        const size_t workspace_bytes = sslm_workspace_size(model, &config);
        workspace_storage.resize(workspace_bytes + 63);
        void* workspace_aligned = workspace_storage.data();
        size_t workspace_space = workspace_storage.size();
        if (!std::align(64, workspace_bytes, workspace_aligned, workspace_space) ||
            sslm_workspace_create(model, &config, workspace_aligned, workspace_bytes, &workspace) !=
                SSLM_OK) {
            return false;
        }
        if (sslm_schema_lookup(model, "t2922_accepts_q", &schema_a) != SSLM_OK ||
            sslm_schema_lookup(model, "t2922_rejects_q", &schema_b) != SSLM_OK) {
            return false;
        }
        if (!ReadFile(adapter_path, &adapter_bytes)) {
            std::fprintf(stderr, "CPU adapter read failed\n");
            return false;
        }
        if (!RetargetAdapter(&adapter_bytes, view.RawIntegrityHash())) return false;
        const sslm_status adapter_status =
            sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), model, &adapter);
        if (adapter_status != SSLM_OK) {
            std::fprintf(stderr, "CPU adapter map failed: %d\n", int(adapter_status));
            return false;
        }
        return true;
    }

    sslm_seq NewSeq() const {
        sslm_seq seq = nullptr;
        return sslm_seq_create(model, const_cast<sslm_kv_pool*>(&pool), &seq) == SSLM_OK ? seq : nullptr;
    }

    std::vector<uint8_t> Save(sslm_seq seq) const {
        size_t size = 0;
        if (sslm_seq_save(seq, nullptr, &size) != SSLM_BUFFER_TOO_SMALL || size == 0) return {};
        std::vector<uint8_t> blob(size);
        if (sslm_seq_save(seq, blob.data(), &size) != SSLM_OK) return {};
        blob.resize(size);
        return blob;
    }

    sslm_status Apply(sslm_seq seq, SchemaOp op) const {
        if (op == SchemaOp::Unbind) return sslm_seq_set_schema(seq, nullptr);
        return sslm_seq_set_schema(seq, op == SchemaOp::Rebind ? schema_b : schema_a);
    }

    bool Prepare(sslm_seq seq, SchemaOp op) const {
        return op == SchemaOp::Bind || sslm_seq_set_schema(seq, schema_a) == SSLM_OK;
    }

    bool Prompt(sslm_seq seq) const {
        const int32_t token = 0;
        int32_t consumed = 0;
        return sslm_prefill(model, seq, &token, 1, 8, SSLM_SPAN_PROMPT, workspace, &consumed) ==
                   SSLM_OK &&
               consumed == 1;
    }

    bool Decode(sslm_seq seq) const {
        if (!Prompt(seq)) return false;
        sslm_decode_params params{};
        params.layer_budget = static_cast<int32_t>(view.config.num_hidden_layers);
        int32_t token = -99;
        return sslm_decode_step(model, &seq, 1, &params, workspace, &token) == SSLM_OK && token >= 0;
    }

    bool DeadEnd(sslm_seq seq) const {
        if (sslm_seq_set_schema(seq, schema_a) != SSLM_OK) return false;
        const int32_t path[] = {0, 1, 2, 5};
        int32_t consumed = 0;
        if (sslm_prefill(model, seq, path, 4, 8, SSLM_SPAN_SCHEMA_CONTENT, workspace, &consumed) !=
                SSLM_OK ||
            consumed != 4) {
            return false;
        }
        sslm_decode_params params{};
        params.layer_budget = static_cast<int32_t>(view.config.num_hidden_layers);
        int32_t token = -99;
        return sslm_decode_step(model, &seq, 1, &params, workspace, &token) == SSLM_OK && token == -2;
    }

    bool Partial(sslm_seq seq) const {
        if (!Prompt(seq)) return false;
        sslm_decode_params full{};
        full.layer_budget = static_cast<int32_t>(view.config.num_hidden_layers);
        int32_t first = -99;
        if (sslm_decode_step(model, &seq, 1, &full, workspace, &first) != SSLM_OK || first < 0) {
            return false;
        }
        sslm_decode_params one{};
        one.layer_budget = 1;
        int32_t partial = -99;
        return sslm_decode_step(model, &seq, 1, &one, workspace, &partial) == SSLM_OK && partial == -1;
    }

    int32_t FindFirstCallGuardToken() const {
        for (int32_t token = 0; token < static_cast<int32_t>(view.config.vocab_size); ++token) {
            sslm_seq seq = NewSeq();
            if (!seq) return -1;
            int32_t consumed = 0;
            const sslm_status status = sslm_prefill(
                model, seq, &token, 1, 8, SSLM_SPAN_PROMPT, workspace, &consumed);
            sslm_seq_release(seq);
            if (status != SSLM_OK && consumed == 0) return token;
        }
        return -1;
    }

    void Close() {
        if (adapter) sslm_adapter_release(adapter);
        if (workspace) sslm_workspace_destroy(workspace);
        if (pool) sslm_kv_pool_destroy(pool);
        if (model) sslm_model_unmap(model);
    }
};

struct GpuEnv {
    SslmGpuContext* ctx = nullptr;
    GpuModelFixture fixture;
    std::vector<uint8_t> adapter_bytes;
    superslm::SslmModelView adapter_view{};
    SslmGpuAdapterHandle* adapter = nullptr;
    int32_t schema_a = -1;
    int32_t schema_b = -1;

    bool Open(const char* model_path, const char* adapter_path) {
        std::string error;
        if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SslmGpuStatus::SSLM_OK || !ctx ||
            !fixture.Open(model_path, ctx)) {
            return false;
        }
        schema_a = SslmGpuSchemaLookupForG5Bridge(fixture.model, "t2922_accepts_q");
        schema_b = SslmGpuSchemaLookupForG5Bridge(fixture.model, "t2922_rejects_q");
        if (schema_a < 0 || schema_b < 0 || !ReadFile(adapter_path, &adapter_bytes) ||
            !RetargetAdapter(&adapter_bytes, fixture.view.RawIntegrityHash()) ||
            superslm::SslmModel::Load(adapter_bytes.data(), adapter_bytes.size(), adapter_view, &error) !=
                superslm::SslmModelStatus::Ok ||
            sslm_gpu_adapter_map(ctx, fixture.model, &adapter_view, &adapter) !=
                SslmGpuStatus::SSLM_OK) {
            std::fprintf(stderr, "GPU adapter open failed: %s\n", error.c_str());
            return false;
        }
        return true;
    }

    SslmGpuSequenceHandle* NewSeq() const {
        SslmGpuSequenceHandle* seq = nullptr;
        return sslm_gpu_seq_create(ctx, fixture.model, fixture.model_cap, &seq) ==
                       SslmGpuStatus::SSLM_OK
                   ? seq
                   : nullptr;
    }

    std::vector<uint8_t> Save(SslmGpuSequenceHandle* seq) const {
        size_t size = 0;
        if (sslm_gpu_seq_save(ctx, seq, nullptr, &size) != SslmGpuStatus::SSLM_DEVICE_LOST ||
            size == 0) {
            return {};
        }
        std::vector<uint8_t> blob(size);
        if (sslm_gpu_seq_save(ctx, seq, blob.data(), &size) != SslmGpuStatus::SSLM_OK) return {};
        blob.resize(size);
        return blob;
    }

    SslmGpuStatus Apply(SslmGpuSequenceHandle* seq, SchemaOp op) const {
        return SslmGpuSeqSetSchemaForG5Bridge(
            ctx, seq, op == SchemaOp::Unbind ? -1 : (op == SchemaOp::Rebind ? schema_b : schema_a));
    }

    bool Prepare(SslmGpuSequenceHandle* seq, SchemaOp op) const {
        return op == SchemaOp::Bind ||
               SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, schema_a) == SslmGpuStatus::SSLM_OK;
    }

    bool Prompt(SslmGpuSequenceHandle* seq) const {
        return Prefill(fixture, seq, {0}) == SslmGpuStatus::SSLM_OK;
    }

    bool Decode(SslmGpuSequenceHandle* seq) const {
        int32_t token = -99;
        return SslmGpuSeqDecodeStepForG5Bridge(
                   ctx, seq, 0, fixture.one_layer_budget, &token) == SslmGpuStatus::SSLM_OK &&
               token >= 0;
    }

    bool DeadEnd(SslmGpuSequenceHandle* seq) const {
        if (SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, schema_a) != SslmGpuStatus::SSLM_OK) return false;
        const int32_t path[] = {0, 1, 2, 5};
        int32_t consumed = 0;
        if (SslmGpuSeqPrefillSchemaContentForG5Bridge(
                ctx, seq, path, 4, fixture.one_layer_budget, &consumed) != SslmGpuStatus::SSLM_OK ||
            consumed != 4) {
            return false;
        }
        int32_t token = -99;
        return SslmGpuSeqDecodeStepForG5Bridge(
                   ctx, seq, 7, fixture.one_layer_budget, &token) == SslmGpuStatus::SSLM_OK &&
               token == -2;
    }

    bool Partial(SslmGpuSequenceHandle* seq) const {
        if (sslm_gpu_seq_embed_token(ctx, seq, 0) != SslmGpuStatus::SSLM_OK ||
            sslm_decode_step_gpu(ctx, seq, nullptr, fixture.one_layer_budget) != SslmGpuStatus::SSLM_OK ||
            Drain(ctx, seq) != SslmGpuStatus::SSLM_OK) {
            return false;
        }
        return LayerIndex(seq) == 1;
    }

    int32_t FindFirstCallGuardToken() const {
        for (int32_t token = 0; token < fixture.vocab; ++token) {
            SslmGpuSequenceHandle* seq = NewSeq();
            if (!seq) return -1;
            const SslmGpuStatus status = Prefill(fixture, seq, {token});
            const bool refused_without_commit =
                status == SslmGpuStatus::SSLM_SEQUENCE_REJECTED && ContextLength(seq) == 0;
            sslm_gpu_seq_release(ctx, seq);
            if (refused_without_commit) return token;
        }
        return -1;
    }

    void Close() {
        if (adapter) sslm_gpu_adapter_unmap(ctx, adapter);
        fixture.Close();
        if (ctx) sslm_gpu_context_destroy(ctx);
    }
};

template <typename Setup>
bool CpuRejectOps(CpuEnv& env, Setup setup, bool restored_copy) {
    bool held = true;
    for (SchemaOp op : {SchemaOp::Bind, SchemaOp::Rebind, SchemaOp::Unbind}) {
        sslm_seq original = env.NewSeq();
        if (!original || !env.Prepare(original, op) || !setup(original)) {
            if (original) sslm_seq_release(original);
            return false;
        }
        sslm_seq target = original;
        if (restored_copy) {
            const std::vector<uint8_t> blob = env.Save(original);
            target = nullptr;
            if (blob.empty() ||
                sslm_seq_restore(env.model, &env.pool, blob.data(), blob.size(), &target) != SSLM_OK) {
                sslm_seq_release(original);
                return false;
            }
        }
        const std::vector<uint8_t> before = env.Save(target);
        const sslm_status status = env.Apply(target, op);
        const std::vector<uint8_t> after = env.Save(target);
        const bool one = status == SSLM_SCHEMA_BIND_REJECTED && !before.empty() && before == after;
        if (!one) std::printf("DETAIL cpu_reject op=%s restored=%d status=%d before=%zu after=%zu equal=%d\n",
                             OpName(op), restored_copy, int(status), before.size(), after.size(), before == after);
        held = held && one;
        if (restored_copy) sslm_seq_release(target);
        sslm_seq_release(original);
    }
    return held;
}

template <typename Setup>
bool GpuRejectOps(GpuEnv& env, Setup setup, bool restored_copy) {
    bool held = true;
    for (SchemaOp op : {SchemaOp::Bind, SchemaOp::Rebind, SchemaOp::Unbind}) {
        SslmGpuSequenceHandle* original = env.NewSeq();
        if (!original || !env.Prepare(original, op) || !setup(original)) {
            if (original) sslm_gpu_seq_release(env.ctx, original);
            return false;
        }
        SslmGpuSequenceHandle* target = original;
        if (restored_copy) {
            const std::vector<uint8_t> blob = env.Save(original);
            target = nullptr;
            if (blob.empty() ||
                sslm_gpu_seq_restore(env.ctx, env.fixture.model, blob.data(), blob.size(), &target) !=
                    SslmGpuStatus::SSLM_OK) {
                sslm_gpu_seq_release(env.ctx, original);
                return false;
            }
        }
        const std::vector<uint8_t> before = env.Save(target);
        const SslmGpuStatus status = env.Apply(target, op);
        const std::vector<uint8_t> after = env.Save(target);
        const bool one = status == SslmGpuStatus::SSLM_SEQUENCE_REJECTED && !before.empty() &&
                         before == after;
        if (!one) std::printf("DETAIL gpu_reject op=%s restored=%d status=%d before=%zu after=%zu equal=%d\n",
                             OpName(op), restored_copy, int(status), before.size(), after.size(), before == after);
        held = held && one;
        if (restored_copy) sslm_gpu_seq_release(env.ctx, target);
        sslm_gpu_seq_release(env.ctx, original);
    }
    return held;
}

bool CpuFreshResetRestoreReset(CpuEnv& env) {
    bool held = true;
    for (SchemaOp op : {SchemaOp::Bind, SchemaOp::Rebind, SchemaOp::Unbind}) {
        sslm_seq fresh = env.NewSeq();
        const bool fresh_ok = fresh && env.Prepare(fresh, op) && env.Apply(fresh, op) == SSLM_OK;
        held = fresh_ok && held;
        if (fresh) sslm_seq_release(fresh);

        sslm_seq reset = env.NewSeq();
        const bool reset_ok = reset && env.Prepare(reset, op) && env.Prompt(reset) &&
                              sslm_seq_reset(reset) == SSLM_OK && env.Apply(reset, op) == SSLM_OK;
        held = reset_ok && held;
        if (reset) sslm_seq_release(reset);

        sslm_seq original = env.NewSeq();
        const bool original_ok = original && env.Prepare(original, op);
        const std::vector<uint8_t> blob = original ? env.Save(original) : std::vector<uint8_t>{};
        sslm_seq restored = nullptr;
        const bool restore_ok = original_ok && !blob.empty() &&
            sslm_seq_restore(env.model, &env.pool, blob.data(), blob.size(), &restored) == SSLM_OK;
        bool restored_ok = false;
        if (restored) {
            const std::vector<uint8_t> before = env.Save(restored);
            const sslm_status before_status = env.Apply(restored, op);
            const bool unchanged = env.Save(restored) == before;
            const sslm_status reset_status = sslm_seq_reset(restored);
            const sslm_status after_status = env.Apply(restored, op);
            restored_ok = restore_ok && before_status == SSLM_SCHEMA_BIND_REJECTED && unchanged &&
                          reset_status == SSLM_OK && after_status == SSLM_OK;
            if (!restored_ok) std::printf("DETAIL cpu_restore op=%s before=%d unchanged=%d reset=%d after=%d\n",
                                          OpName(op), int(before_status), unchanged, int(reset_status), int(after_status));
            sslm_seq_release(restored);
        }
        held = restored_ok && held;
        if (original) sslm_seq_release(original);
    }
    return held;
}

bool GpuFreshResetRestoreReset(GpuEnv& env) {
    bool held = true;
    for (SchemaOp op : {SchemaOp::Bind, SchemaOp::Rebind, SchemaOp::Unbind}) {
        SslmGpuSequenceHandle* fresh = env.NewSeq();
        const bool fresh_ok = fresh && env.Prepare(fresh, op) &&
                              env.Apply(fresh, op) == SslmGpuStatus::SSLM_OK;
        held = fresh_ok && held;
        if (fresh) sslm_gpu_seq_release(env.ctx, fresh);

        SslmGpuSequenceHandle* reset = env.NewSeq();
        const bool reset_ok = reset && env.Prepare(reset, op) && env.Prompt(reset) &&
            sslm_gpu_seq_reset(env.ctx, reset) == SslmGpuStatus::SSLM_OK &&
            env.Apply(reset, op) == SslmGpuStatus::SSLM_OK;
        held = reset_ok && held;
        if (reset) sslm_gpu_seq_release(env.ctx, reset);

        SslmGpuSequenceHandle* original = env.NewSeq();
        const bool original_ok = original && env.Prepare(original, op);
        const std::vector<uint8_t> blob = original ? env.Save(original) : std::vector<uint8_t>{};
        SslmGpuSequenceHandle* restored = nullptr;
        const bool restore_ok = original_ok && !blob.empty() &&
            sslm_gpu_seq_restore(env.ctx, env.fixture.model, blob.data(), blob.size(), &restored) ==
                SslmGpuStatus::SSLM_OK;
        bool restored_ok = false;
        if (restored) {
            const std::vector<uint8_t> before = env.Save(restored);
            const SslmGpuStatus before_status = env.Apply(restored, op);
            const bool unchanged = env.Save(restored) == before;
            const SslmGpuStatus reset_status = sslm_gpu_seq_reset(env.ctx, restored);
            const SslmGpuStatus after_status = env.Apply(restored, op);
            restored_ok = restore_ok && before_status == SslmGpuStatus::SSLM_SEQUENCE_REJECTED &&
                          unchanged && reset_status == SslmGpuStatus::SSLM_OK &&
                          after_status == SslmGpuStatus::SSLM_OK;
            if (!restored_ok) std::printf("DETAIL gpu_restore op=%s before=%d unchanged=%d reset=%d after=%d\n",
                                          OpName(op), int(before_status), unchanged, int(reset_status), int(after_status));
            sslm_gpu_seq_release(env.ctx, restored);
        }
        held = restored_ok && held;
        if (original) sslm_gpu_seq_release(env.ctx, original);
    }
    return held;
}

bool CpuResetReuse(CpuEnv& env) {
    sslm_seq dead = env.NewSeq();
    const bool dead_ready = dead && env.DeadEnd(dead);
    const sslm_status dead_reset = dead_ready ? sslm_seq_reset(dead) : SSLM_INVALID_ARGUMENT;
    const bool dead_decode = dead_reset == SSLM_OK && env.Decode(dead);
    bool held = dead_ready && dead_reset == SSLM_OK && dead_decode;
    if (dead) sslm_seq_release(dead);
    sslm_seq partial = env.NewSeq();
    const bool partial_ready = partial && env.Partial(partial);
    const sslm_status partial_reset = partial_ready ? sslm_seq_reset(partial) : SSLM_INVALID_ARGUMENT;
    const bool partial_decode = partial_reset == SSLM_OK && env.Decode(partial);
    held = held && partial_ready && partial_reset == SSLM_OK && partial_decode;
    if (!held) std::printf("DETAIL cpu_reset_reuse dead_ready=%d dead_reset=%d dead_decode=%d partial_ready=%d partial_reset=%d partial_decode=%d\n",
                           dead_ready, int(dead_reset), dead_decode, partial_ready, int(partial_reset), partial_decode);
    if (partial) sslm_seq_release(partial);
    return held;
}

bool GpuResetReuse(GpuEnv& env) {
    SslmGpuSequenceHandle* dead = env.NewSeq();
    const bool dead_ready = dead && env.DeadEnd(dead);
    const SslmGpuStatus dead_reset = dead_ready ? sslm_gpu_seq_reset(env.ctx, dead) : SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
    const bool dead_decode = dead_reset == SslmGpuStatus::SSLM_OK && env.Decode(dead);
    bool held = dead_ready && dead_reset == SslmGpuStatus::SSLM_OK && dead_decode;
    if (dead) sslm_gpu_seq_release(env.ctx, dead);
    SslmGpuSequenceHandle* partial = env.NewSeq();
    const bool partial_ready = partial && env.Partial(partial);
    const SslmGpuStatus partial_reset = partial_ready ? sslm_gpu_seq_reset(env.ctx, partial) : SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
    const bool partial_decode = partial_reset == SslmGpuStatus::SSLM_OK && env.Decode(partial);
    held = held && partial_ready && partial_reset == SslmGpuStatus::SSLM_OK && partial_decode;
    if (!held) std::printf("DETAIL gpu_reset_reuse dead_ready=%d dead_reset=%d dead_decode=%d partial_ready=%d partial_reset=%d partial_decode=%d\n",
                           dead_ready, int(dead_reset), dead_decode, partial_ready, int(partial_reset), partial_decode);
    if (partial) sslm_gpu_seq_release(env.ctx, partial);
    return held;
}

bool CpuAdapterReset(CpuEnv& env) {
    sslm_seq seq = env.NewSeq();
    const bool partial = seq && env.Partial(seq);
    const sslm_status refused = partial ? sslm_seq_set_adapter(seq, env.adapter) : SSLM_INVALID_ARGUMENT;
    const sslm_status reset = partial ? sslm_seq_reset(seq) : SSLM_INVALID_ARGUMENT;
    const sslm_status accepted = reset == SSLM_OK ? sslm_seq_set_adapter(seq, env.adapter) : SSLM_INVALID_ARGUMENT;
    const bool held = partial && refused == SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED && reset == SSLM_OK && accepted == SSLM_OK;
    if (!held) std::printf("DETAIL cpu_adapter partial=%d refused=%d reset=%d accepted=%d\n", partial, int(refused), int(reset), int(accepted));
    if (seq) sslm_seq_release(seq);
    return held;
}

bool GpuAdapterReset(GpuEnv& env) {
    SslmGpuSequenceHandle* seq = env.NewSeq();
    const bool partial = seq && env.Partial(seq);
    const SslmGpuStatus refused = partial ? sslm_gpu_seq_bind_adapter(env.ctx, seq, env.adapter) : SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
    const SslmGpuStatus reset = partial ? sslm_gpu_seq_reset(env.ctx, seq) : SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
    const SslmGpuStatus accepted = reset == SslmGpuStatus::SSLM_OK ? sslm_gpu_seq_bind_adapter(env.ctx, seq, env.adapter) : SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
    const bool held = partial && refused == SslmGpuStatus::SSLM_BUSY && reset == SslmGpuStatus::SSLM_OK && accepted == SslmGpuStatus::SSLM_OK;
    if (!held) std::printf("DETAIL gpu_adapter partial=%d refused=%d reset=%d accepted=%d\n", partial, int(refused), int(reset), int(accepted));
    if (seq) sslm_gpu_seq_release(env.ctx, seq);
    return held;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <dual.sslm> <guard.sslm> <adapter.sslm>\n", argv[0]);
        return 2;
    }
    CpuEnv cpu;
    CpuEnv cpu_guard;
    GpuEnv gpu;
    GpuEnv gpu_guard;
    if (!cpu.Open(argv[1], argv[3]) || !cpu_guard.Open(argv[2], argv[3]) ||
        !gpu.Open(argv[1], argv[3]) || !gpu_guard.Open(argv[2], argv[3])) {
        std::fprintf(stderr, "fixture setup failed\n");
        return 2;
    }

    const bool cpu_fresh_reset_restore = CpuFreshResetRestoreReset(cpu);
    const bool gpu_fresh_reset_restore = GpuFreshResetRestoreReset(gpu);
    Cell("BIND_FRESH_RESET_RESTORED_RESET_ACCEPT",
         cpu_fresh_reset_restore && gpu_fresh_reset_restore,
         "create/reset accept; restore rejects unchanged until reset; CPU+GPU bind/rebind/unbind");

    const bool cpu_prompt_live = CpuRejectOps(cpu, [&](sslm_seq seq) { return cpu.Prompt(seq); }, false);
    const bool cpu_prompt_restored = CpuRejectOps(cpu, [&](sslm_seq seq) { return cpu.Prompt(seq); }, true);
    const bool gpu_prompt_live = GpuRejectOps(gpu, [&](SslmGpuSequenceHandle* seq) { return gpu.Prompt(seq); }, false);
    const bool gpu_prompt_restored = GpuRejectOps(gpu, [&](SslmGpuSequenceHandle* seq) { return gpu.Prompt(seq); }, true);
    const bool prompt = cpu_prompt_live && cpu_prompt_restored && gpu_prompt_live && gpu_prompt_restored;
    Cell("BIND_AFTER_PROMPT_PREFILL_REJECT", prompt,
         "live and SLM5-restored prompt histories reject unchanged on both backends");

    const bool cpu_decode_reject = CpuRejectOps(cpu, [&](sslm_seq seq) { return cpu.Decode(seq); }, false);
    const bool gpu_decode_reject = GpuRejectOps(gpu, [&](SslmGpuSequenceHandle* seq) { return gpu.Decode(seq); }, false);
    Cell("BIND_AFTER_DECODE_REJECT", cpu_decode_reject && gpu_decode_reject,
         "normal decoded-token histories reject unchanged on both backends");

    const bool cpu_dead_reject = CpuRejectOps(cpu, [&](sslm_seq seq) { return cpu.DeadEnd(seq); }, false);
    const bool gpu_dead_reject = GpuRejectOps(gpu, [&](SslmGpuSequenceHandle* seq) { return gpu.DeadEnd(seq); }, false);
    Cell("BIND_AFTER_DEAD_END_REJECT", cpu_dead_reject && gpu_dead_reject,
         "TE-377 dead-end histories reject unchanged before reset on both backends");

    const int32_t cpu_guard_token = cpu_guard.FindFirstCallGuardToken();
    const int32_t gpu_guard_token = gpu_guard.FindFirstCallGuardToken();
    const bool cpu_guard_reject = cpu_guard_token >= 0 && CpuRejectOps(cpu_guard, [&](sslm_seq seq) {
            int32_t consumed = 0;
            return sslm_prefill(cpu_guard.model, seq, &cpu_guard_token, 1, 8, SSLM_SPAN_PROMPT,
                                cpu_guard.workspace, &consumed) != SSLM_OK && consumed == 0;
        }, false);
    const bool gpu_guard_reject = gpu_guard_token >= 0 && GpuRejectOps(gpu_guard, [&](SslmGpuSequenceHandle* seq) {
            return Prefill(gpu_guard.fixture, seq, {gpu_guard_token}) ==
                       SslmGpuStatus::SSLM_SEQUENCE_REJECTED &&
                   ContextLength(seq) == 0;
        }, false);
    const bool guard_reject = cpu_guard_reject && gpu_guard_reject;
    Cell("BIND_AFTER_REFUSED_FIRST_CALL_REJECT", guard_reject,
         "one documented first-call prompt guard refusal clears eligibility on CPU+GPU");

    const bool cpu_reset_reuse = CpuResetReuse(cpu);
    const bool gpu_reset_reuse = GpuResetReuse(gpu);
    Cell("RESET_REUSE", cpu_reset_reuse && gpu_reset_reuse,
         "dead-end and ordinary throttled partial-token histories reset and decode a real token");

    bool guard_reset = false;
    if (cpu_guard_token >= 0 && gpu_guard_token >= 0) {
        sslm_seq cpu_seq = cpu_guard.NewSeq();
        int32_t consumed = 0;
        const bool cpu_refused = cpu_seq &&
            sslm_prefill(cpu_guard.model, cpu_seq, &cpu_guard_token, 1, 8, SSLM_SPAN_PROMPT,
                         cpu_guard.workspace, &consumed) != SSLM_OK && consumed == 0;
        consumed = 0;
        const bool cpu_reused = cpu_refused && sslm_seq_reset(cpu_seq) == SSLM_OK &&
            sslm_prefill(cpu_guard.model, cpu_seq, &cpu_guard_token, 1, 8, SSLM_SPAN_PROMPT,
                         cpu_guard.workspace, &consumed) != SSLM_OK && consumed == 0;
        if (cpu_seq) sslm_seq_release(cpu_seq);

        SslmGpuSequenceHandle* gpu_seq = gpu_guard.NewSeq();
        const bool gpu_refused = gpu_seq &&
            Prefill(gpu_guard.fixture, gpu_seq, {gpu_guard_token}) ==
                SslmGpuStatus::SSLM_SEQUENCE_REJECTED &&
            ContextLength(gpu_seq) == 0;
        const bool gpu_reused = gpu_refused &&
            sslm_gpu_seq_reset(gpu_guard.ctx, gpu_seq) == SslmGpuStatus::SSLM_OK &&
            Prefill(gpu_guard.fixture, gpu_seq, {gpu_guard_token}) ==
                SslmGpuStatus::SSLM_SEQUENCE_REJECTED &&
            ContextLength(gpu_seq) == 0;
        if (gpu_seq) sslm_gpu_seq_release(gpu_guard.ctx, gpu_seq);
        guard_reset = cpu_reused && gpu_reused;
        if (!guard_reset) std::printf("DETAIL guard_reset cpu_refused=%d cpu_reused=%d gpu_refused=%d gpu_reused=%d tokens=%d/%d\n",
                                     cpu_refused, cpu_reused, gpu_refused, gpu_reused,
                                     cpu_guard_token, gpu_guard_token);
    }
    Cell("RESET_AFTER_GUARD_REFUSAL", guard_reset,
         "one real first-call guard refusal resets and re-enters the caller, reproducing the guard instead of a reset/busy refusal");

    const bool cpu_adapter_reset = CpuAdapterReset(cpu);
    const bool gpu_adapter_reset = GpuAdapterReset(gpu);
    Cell("ADAPTER_SWAP_RESET", cpu_adapter_reset && gpu_adapter_reset,
         "real mapped adapter swap rejects at throttled partial token and succeeds after reset");

    cpu.Close();
    cpu_guard.Close();
    gpu.Close();
    gpu_guard.Close();
    std::printf("SUMMARY checks=%d failures=%d skips=0\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
