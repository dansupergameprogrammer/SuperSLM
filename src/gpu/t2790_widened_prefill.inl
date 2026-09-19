// T-2790 spike: the row-widened GPU prompt prefill (TE-266 plan §5.2/§5.3). Included at the end of
// superslm_gpu.cpp, inside namespace superslm_gpu, only when SUPERSLM_T2790_WIDE_H is defined -- no
// shipped target defines it. Scope, as the spike is specified: the prompt path only, no adapters, no
// schema twin, no guard-replay path. A device-side guard refusal in a widened chunk is reported, not
// replayed.
//
// Shape. A chunk is SUPERSLM_T2790_WIDE_H consecutive prompt tokens ("rows"). Each layer's 25 sites
// are recorded ONCE per chunk instead of once per token:
//   - the 19 non-GEMM sites run the generated widened shaders (shaders_t2790/*_wide.hlsl), dispatched
//     (groups_x, H, 1): group row y is token row y, with its own SeqState/LayerScratch/WorkScratch
//     window and its own position (row 0's position + y);
//   - the six projection GEMMs run proj_gemm_wide.hlsl, one pass over each weight row serving all H
//     rows.
// Rows share the sequence's one K/V buffer. Every row writes only its own position; attention row r
// reads positions 0..p_r, all committed by earlier dispatches (earlier rows of this chunk included),
// and never a later row's.
//
// Fixed height: every dispatch is shaped for H rows whatever the document length. A short final
// chunk is padded with inert rows -- their sticky word is set before the chunk runs, so every site
// returns before touching anything (no K/V write, no commit, no context advance) and the host
// discards them.
#if defined(SUPERSLM_T2790_WIDE_H)

namespace {

constexpr uint32_t kT2790Rows = SUPERSLM_T2790_WIDE_H;
static_assert(kT2790Rows >= 1 && kT2790Rows <= 64, "T-2790 spike rows out of range");
static_assert(kT2790Rows < 8 || (kT2790Rows % 8) == 0, "proj_gemm_wide.hlsl reduces in batches of 8 rows");
constexpr int64_t kT2790InertRowTag = 99;  // any nonzero sticky word; never decoded
constexpr uint32_t kT2790SitesPerLayer = 25;

uint64_t T2790Align256(uint64_t x) { return (x + 255u) & ~uint64_t{255}; }

struct T2790Resources {
	Microsoft::WRL::ComPtr<ID3D12RootSignature> root_sig;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
	Microsoft::WRL::ComPtr<ID3D12Resource> seq_rows, scratch_rows, work_rows;
	uint64_t seq_stride = 0, scratch_stride = 0, work_stride = 0;
	Microsoft::WRL::ComPtr<ID3D12Resource> seq_readback;
	Microsoft::WRL::ComPtr<ID3D12Resource> kv_readback;
	uint64_t kv_readback_bytes = 0;
};
T2790Resources g_t2790;
bool g_t2790_enabled = false;

// Timing, accumulated across chunks until reset: per-site GPU milliseconds (the timestamp delta
// before each dispatch to the next, summed over layers), host recording and fence-wait time.
double g_t2790_site_ms[kT2790SitesPerLayer] = {};
uint64_t g_t2790_chunks = 0;
double g_t2790_record_ms = 0, g_t2790_wait_ms = 0, g_t2790_gpu_busy_ms = 0;

Microsoft::WRL::ComPtr<ID3D12RootSignature> T2790MakeRootSig(harness::Device& dev) {
	D3D12_DESCRIPTOR_RANGE ranges[3]{};
	for (uint32_t i = 0; i < 3; ++i) {
		ranges[i].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		ranges[i].NumDescriptors = kT2790Rows;
		ranges[i].BaseShaderRegister = 0;
		ranges[i].RegisterSpace = i + 1;  // space1 SeqState, space2 LayerScratch, space3 WorkScratch
		ranges[i].OffsetInDescriptorsFromTableStart = i * kT2790Rows;
	}
	D3D12_ROOT_PARAMETER ps[13]{};
	ps[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	ps[0].Constants.Num32BitValues = 28;
	ps[0].Constants.ShaderRegister = 0;
	ps[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	for (uint32_t i = 0; i < 8; ++i) {  // t0..t7, the composed signature's own eight SRVs
		ps[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		ps[1 + i].Descriptor.ShaderRegister = i;
		ps[1 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	}
	ps[9].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	ps[9].DescriptorTable.NumDescriptorRanges = 3;
	ps[9].DescriptorTable.pDescriptorRanges = ranges;
	ps[9].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	ps[10].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;  // u2 KvCache
	ps[10].Descriptor.ShaderRegister = 2;
	ps[10].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	ps[11].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;  // t8 LoraAB (declared by the tail sites; unread at rank 0)
	ps[11].Descriptor.ShaderRegister = 8;
	ps[11].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	ps[12].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;  // t9 Fold
	ps[12].Descriptor.ShaderRegister = 9;
	ps[12].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	D3D12_ROOT_SIGNATURE_DESC rs{};
	rs.NumParameters = 13;
	rs.pParameters = ps;
	Microsoft::WRL::ComPtr<ID3DBlob> blob, err;
	if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) {
		if (err) std::fprintf(stderr, "%s\n", static_cast<const char*>(err->GetBufferPointer()));
		throw std::runtime_error("T-2790 widened root signature serialization failed");
	}
	Microsoft::WRL::ComPtr<ID3D12RootSignature> r;
	SSLM_GPU_HR(dev.dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
	                                          IID_PPV_ARGS(&r)));
	return r;
}

// The 25 per-layer sites, in recording order; index = site id used for timing.
const char* const kT2790SiteShader[kT2790SitesPerLayer] = {
    "attn_norm_site_wide",     "q_proj_gemm_wide",        "q_proj_site_wide",
    "kv_proj_gemm_wide",       "kv_proj_site_wide",       "qk_norm_site_wide",
    "rope_guard_site_wide",    "rope_commit_site_wide",   "attention_score_site_wide",
    "softmax_site_wide",       "context_accumulate_site_wide", "ctx_fold_site_wide",
    "o_proj_gemm_wide",        "o_proj_site_wide",        "attn_residual_site_wide",
    "mlp_norm_site_wide",      "gate_proj_gemm_wide",     "gate_proj_site_wide",
    "up_proj_gemm_wide",       "up_proj_site_wide",       "mlp_act_site_wide",
    "down_proj_gemm_wide",     "down_proj_site_wide",     "mlp_residual_site_wide",
    "commit_site_wide"};
Microsoft::WRL::ComPtr<ID3D12PipelineState> g_t2790_pso_by_site[kT2790SitesPerLayer];

// Resolved once per process: no per-dispatch lookup in the recording loop.
void T2790EnsurePsos(harness::Device& dev) {
	if (g_t2790_pso_by_site[0]) return;
	for (uint32_t s = 0; s < kT2790SitesPerLayer; ++s) {
		const std::string name = std::string(kT2790SiteShader[s]) + "_h" + std::to_string(kT2790Rows);
		auto cso = harness::ReadFile(harness::ShaderPath(name));
		g_t2790_pso_by_site[s] = dev.MakePSO(g_t2790.root_sig.Get(), cso);
	}
}

// (Re)creates the three H-row buffers and their descriptors when a row outgrows them.
void T2790EnsureRowBuffers(harness::Device& dev, uint64_t seq_bytes, uint64_t scratch_bytes,
                           uint64_t work_bytes) {
	const uint64_t seq_stride = T2790Align256(seq_bytes);
	const uint64_t scratch_stride = T2790Align256(scratch_bytes);
	const uint64_t work_stride = T2790Align256(work_bytes);
	if (g_t2790.seq_rows && seq_stride <= g_t2790.seq_stride && scratch_stride <= g_t2790.scratch_stride &&
	    work_stride <= g_t2790.work_stride) {
		return;
	}
	g_t2790.seq_stride = std::max(seq_stride, g_t2790.seq_stride);
	g_t2790.scratch_stride = std::max(scratch_stride, g_t2790.scratch_stride);
	g_t2790.work_stride = std::max(work_stride, g_t2790.work_stride);
	// SeqState rows rest in COPY_DEST between chunks (the next chunk's initial state is copied in).
	g_t2790.seq_rows = dev.MakeBuffer(g_t2790.seq_stride * kT2790Rows, D3D12_HEAP_TYPE_DEFAULT,
	                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
	g_t2790.scratch_rows = dev.MakeBuffer(g_t2790.scratch_stride * kT2790Rows, D3D12_HEAP_TYPE_DEFAULT,
	                                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
	                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	g_t2790.work_rows = dev.MakeBuffer(g_t2790.work_stride * kT2790Rows, D3D12_HEAP_TYPE_DEFAULT,
	                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
	                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	g_t2790.seq_readback = dev.MakeBuffer(g_t2790.seq_stride * kT2790Rows, D3D12_HEAP_TYPE_READBACK,
	                                      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
	if (!g_t2790.heap) {
		D3D12_DESCRIPTOR_HEAP_DESC hd{};
		hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		hd.NumDescriptors = 3 * kT2790Rows;
		hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		SSLM_GPU_HR(dev.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_t2790.heap)));
	}
	const UINT inc = dev.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	D3D12_CPU_DESCRIPTOR_HANDLE h = g_t2790.heap->GetCPUDescriptorHandleForHeapStart();
	ID3D12Resource* bufs[3] = {g_t2790.seq_rows.Get(), g_t2790.scratch_rows.Get(), g_t2790.work_rows.Get()};
	const uint64_t strides[3] = {g_t2790.seq_stride, g_t2790.scratch_stride, g_t2790.work_stride};
	for (uint32_t k = 0; k < 3; ++k) {
		for (uint32_t r = 0; r < kT2790Rows; ++r) {
			D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
			ud.Format = DXGI_FORMAT_R32_TYPELESS;
			ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			ud.Buffer.FirstElement = (strides[k] * r) / 4u;
			ud.Buffer.NumElements = static_cast<UINT>(strides[k] / 4u);
			ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
			D3D12_CPU_DESCRIPTOR_HANDLE hh = h;
			hh.ptr += static_cast<SIZE_T>(k * kT2790Rows + r) * inc;
			dev.dev->CreateUnorderedAccessView(bufs[k], nullptr, &ud, hh);
		}
	}
}

}  // namespace

void T2790SetWidenedEnabled(bool enabled) { g_t2790_enabled = enabled; }
bool T2790WidenedEnabled() { return g_t2790_enabled; }
uint32_t T2790Rows() { return kT2790Rows; }
void T2790ResetTimings() {
	for (double& v : g_t2790_site_ms) v = 0;
	g_t2790_chunks = 0;
	g_t2790_record_ms = g_t2790_wait_ms = g_t2790_gpu_busy_ms = 0;
}
void T2790GetTimings(double* site_ms_25, uint64_t* chunks, double* record_ms, double* wait_ms,
                     double* gpu_busy_ms) {
	for (uint32_t i = 0; i < kT2790SitesPerLayer; ++i) site_ms_25[i] = g_t2790_site_ms[i];
	*chunks = g_t2790_chunks;
	*record_ms = g_t2790_record_ms;
	*wait_ms = g_t2790_wait_ms;
	*gpu_busy_ms = g_t2790_gpu_busy_ms;
}

// Drives `chunk_len` pre-scanned, admitted prompt tokens to full depth in chunks of kT2790Rows rows.
// Same arguments and the same host-visible outcome as SubmitChunkToFullDepthForG5Bridge followed by
// RunLayerLoopGpuFinish: on success `seq` holds the last token's final-layer residual, scale, layer
// index and context length, the saturation counters include every row's increments, and the host K/V
// mirror holds every committed row. On a device-side guard refusal it returns the decoded status with
// `seq.context_length` at the failing chunk's start (the caller's derived count then reports the
// shortfall); the spike has no replay path.
superslm::SslmForwardStatus T2790SubmitWidenedChunk(
    superslm::SequenceLayerState& seq, uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim,
    size_t num_key_value_heads, size_t intermediate_size, int64_t context_cap, uint8_t* workspace,
    size_t workspace_size, const uint8_t* chunk_embedding_bytes, uint32_t chunk_len,
    ID3D12Resource* external_kv_resident, bool* io_external_kv_needs_resume_barrier,
    ID3D12Resource* external_weights_resident, ID3D12Resource* external_rope_cos_resident,
    ID3D12Resource* external_rope_sin_resident, bool external_rope_has, uint64_t external_rope_cos_elems,
    uint64_t external_rope_sin_elems, size_t q_width, uint64_t model_generation, bool model_has_qk_norm) {
	static const superslm::SslmTensorManifest kEmptyManifest;
	harness::Device& dev = harness::GetDevice();
	const uint32_t N = num_hidden_layers;
	const uint32_t H = static_cast<uint32_t>(hidden_size);
	const size_t embed_block_bytes = static_cast<size_t>(SeqScaleOff(H)) + 16u;
	uint32_t done = 0;
	while (done < chunk_len) {
		const uint32_t n_real = std::min(kT2790Rows, chunk_len - done);
		const int64_t p0 = seq.context_length;
		seq.layer_index = 0;  // as SubmitChunkToFullDepthForG5Bridge does between sub-chunks
		GpuLayerLoopChunkOpenState state;
		Microsoft::WRL::ComPtr<ID3D12Resource> init_upload;
		std::vector<uint32_t> dispatch_site;
		const auto t_record_start = std::chrono::steady_clock::now();
		try {
			const superslm::SslmForwardStatus prep = PrepareGpuLayerLoopChunkOpenState(
			    seq, /*layers=*/nullptr, N, /*layer_budget=*/N, hidden_size, head_dim, num_key_value_heads,
			    intermediate_size, context_cap, kEmptyManifest, workspace, workspace_size, external_kv_resident,
			    io_external_kv_needs_resume_barrier, external_weights_resident, external_rope_cos_resident,
			    external_rope_sin_resident, external_rope_has, external_rope_cos_elems, external_rope_sin_elems,
			    /*adapter_bridge=*/nullptr, &state, q_width, model_generation);
			if (prep != superslm::SslmForwardStatus::Ok) return prep;
			const uint32_t HD = state.HD, NH = state.NH, NQH = state.NQH, I = state.I;
			const uint32_t cap = state.context_cap_u32;
			const uint32_t Q_WIDTH = NQH * HD;

			if (!g_t2790.root_sig) g_t2790.root_sig = T2790MakeRootSig(dev);
			T2790EnsurePsos(dev);
			T2790EnsureRowBuffers(dev, SeqTotalSize(H), state.scratch_uav->GetDesc().Width,
			                      state.work_scratch_uav->GetDesc().Width);

			// Each row's initial SeqState: its token's embedding block, layer 0, zero counters, its
			// own context length, sticky Ok. Padded rows: sticky set, so every site skips them.
			std::vector<uint8_t> init(static_cast<size_t>(g_t2790.seq_stride) * kT2790Rows, 0);
			for (uint32_t r = 0; r < kT2790Rows; ++r) {
				std::vector<uint8_t> row(SeqTotalSize(H), 0);
				if (r < n_real) {
					std::memcpy(row.data(), chunk_embedding_bytes + static_cast<size_t>(done + r) * embed_block_bytes,
					            embed_block_bytes);
				}
				PutI64At(row, SeqCtxLenOff(H), p0 + r);
				PutI64At(row, SeqStickyOff(H), r < n_real ? 0 : kT2790InertRowTag);
				std::memcpy(init.data() + static_cast<size_t>(g_t2790.seq_stride) * r, row.data(), row.size());
			}
			init_upload = dev.Upload(init.data(), init.size());
			dev.list->CopyBufferRegion(g_t2790.seq_rows.Get(), 0, init_upload.Get(), 0, init.size());
			D3D12_RESOURCE_BARRIER to_uav{};
			to_uav.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			to_uav.Transition.pResource = g_t2790.seq_rows.Get();
			to_uav.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
			to_uav.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
			to_uav.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
			dev.list->ResourceBarrier(1, &to_uav);

			// Bindings: the composed signature's resources, the three per-row arrays as one table.
			dev.list->SetComputeRootSignature(g_t2790.root_sig.Get());
			ID3D12DescriptorHeap* heaps[] = {g_t2790.heap.Get()};
			dev.list->SetDescriptorHeaps(1, heaps);
			dev.list->SetComputeRootShaderResourceView(1, external_weights_resident->GetGPUVirtualAddress());
			dev.list->SetComputeRootShaderResourceView(2, state.layout_buf->GetGPUVirtualAddress());
			dev.list->SetComputeRootShaderResourceView(3, state.rope_buf->GetGPUVirtualAddress());
			dev.list->SetComputeRootShaderResourceView(4, state.model_const_buf->GetGPUVirtualAddress());
			dev.list->SetComputeRootShaderResourceView(5, state.silu_lut_buf->GetGPUVirtualAddress());
			dev.list->SetComputeRootShaderResourceView(6, external_rope_cos_resident->GetGPUVirtualAddress());
			dev.list->SetComputeRootShaderResourceView(7, external_rope_sin_resident->GetGPUVirtualAddress());
			dev.list->SetComputeRootShaderResourceView(8, state.scratch_layout_buf->GetGPUVirtualAddress());
			dev.list->SetComputeRootDescriptorTable(9, g_t2790.heap->GetGPUDescriptorHandleForHeapStart());
			dev.list->SetComputeRootUnorderedAccessView(10, state.kv_uav->GetGPUVirtualAddress());
			dev.list->SetComputeRootShaderResourceView(11, state.layout_buf->GetGPUVirtualAddress());
			dev.list->SetComputeRootShaderResourceView(12, state.layout_buf->GetGPUVirtualAddress());

			D3D12_RESOURCE_BARRIER uav_barrier{};
			uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
			uav_barrier.UAV.pResource = nullptr;
			const uint32_t position0 = static_cast<uint32_t>(p0);
			const uint32_t width0 = position0 + 1u;
			uint32_t qi = 0;
			auto stamp = [&](uint32_t site) {
				if (qi < harness::Device::kMaxTimestampSlots - 1) {
					dev.list->EndQuery(dev.timestamp_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, qi);
				}
				++qi;
				dispatch_site.push_back(site);
			};
			// Plain sites: the 12-value block bind_and_dispatch writes. Rows go in Y unless `gemm`.
			auto plain = [&](uint32_t site, uint32_t layer, uint32_t groups_x, uint32_t lanes,
			                 bool gemm) {
				stamp(site);
				uint32_t consts[12] = {layer, H, HD, NH, cap, position0, NQH, width0, I, N, lanes, Q_WIDTH};
				dev.list->SetComputeRoot32BitConstants(0, 12, consts, 0);
				dev.list->SetPipelineState(g_t2790_pso_by_site[site].Get());
				dev.list->Dispatch(groups_x, gemm ? 1u : kT2790Rows, 1);
				dev.list->ResourceBarrier(1, &uav_barrier);
			};
			// Tail sites: the 28-value block bind_and_dispatch_tail writes with no adapter bound.
			struct Slot { uint32_t in_base, wide_base; };
			auto tail = [&](uint32_t site, uint32_t layer, std::initializer_list<Slot> slots) {
				stamp(site);
				uint32_t consts[28] = {layer, H, HD, NH, cap, position0, NQH, width0, I, N, 1u};
				size_t base = 11;
				for (const Slot& s : slots) {
					consts[base + 0] = 0;  // rank: no adapter
					consts[base + 1] = 0;
					consts[base + 2] = 0;
					consts[base + 3] = 0;
					consts[base + 4] = static_cast<uint32_t>(state.work_adapter_u_off);
					consts[base + 5] = s.in_base;
					consts[base + 6] = s.wide_base;
					consts[base + 7] = Stage1LanesForRank(0);
					base += 8;
				}
				consts[27] = Q_WIDTH;
				dev.list->SetComputeRoot32BitConstants(0, 28, consts, 0);
				dev.list->SetPipelineState(g_t2790_pso_by_site[site].Get());
				dev.list->Dispatch(1, kT2790Rows, 1);
				dev.list->ResourceBarrier(1, &uav_barrier);
			};

			const uint32_t KV2 = 2u * NH * HD;
			const GpuGemmSiteGroupPlan q_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::QProj, H, KV2, I, Q_WIDTH);
			const GpuGemmSiteGroupPlan o_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::OProj, H, KV2, I);
			const GpuGemmSiteGroupPlan kv_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::KvProj, H, KV2, I);
			const GpuGemmSiteGroupPlan gate_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::GateProj, H, KV2, I);
			const GpuGemmSiteGroupPlan up_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::UpProj, H, KV2, I);
			const GpuGemmSiteGroupPlan down_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::DownProj, H, KV2, I);
			// Attention covers the widest row of the chunk; narrower rows' surplus threads return
			// at the shader's own `t >= items` check.
			const uint32_t max_width = position0 + kT2790Rows;
			const uint32_t attn_score_groups = (NQH * max_width + 255u) / 256u;
			const uint32_t ctx_accum_groups = (NQH * HD + 255u) / 256u;
			const uint32_t rope_groups = (NQH * (HD / 2u) + 255u) / 256u;
			const GpuScratchLayout& sl = state.scratch_layout;
			const uint32_t wa = static_cast<uint32_t>(state.work_wide_a_off);
			const uint32_t wb = static_cast<uint32_t>(state.work_wide_b_off);

			for (uint32_t l = 0; l < N; ++l) {
				plain(0, l, 1, 1, false);
				plain(1, l, q_plan.groups, q_plan.lanes, true);
				tail(2, l, {{sl.normed, wa}});
				plain(3, l, kv_plan.groups, kv_plan.lanes, true);
				tail(4, l, {{sl.normed, wa}, {sl.normed, wb}});
				if (model_has_qk_norm) plain(5, l, NQH + NH, 1, false);
				plain(6, l, rope_groups, 1, false);
				plain(7, l, rope_groups, 1, false);
				plain(8, l, attn_score_groups, 1, false);
				plain(9, l, 1, 1, false);
				plain(10, l, ctx_accum_groups, 1, false);
				plain(11, l, 1, 1, false);
				plain(12, l, o_plan.groups, o_plan.lanes, true);
				tail(13, l, {{sl.ctx_codes, wa}});
				plain(14, l, 1, 1, false);
				plain(15, l, 1, 1, false);
				plain(16, l, gate_plan.groups, gate_plan.lanes, true);
				tail(17, l, {{sl.normed, wa}});
				plain(18, l, up_plan.groups, up_plan.lanes, true);
				tail(19, l, {{sl.normed, wa}});
				plain(20, l, 1, 1, false);
				plain(21, l, down_plan.groups, down_plan.lanes, true);
				tail(22, l, {{sl.act_codes, wa}});
				plain(23, l, 1, 1, false);
				plain(24, l, 1, 1, false);
			}
			const uint32_t n_stamps = std::min(qi, harness::Device::kMaxTimestampSlots - 1);
			dev.list->EndQuery(dev.timestamp_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, n_stamps);
			dev.list->ResolveQueryData(dev.timestamp_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, n_stamps + 1,
			                           dev.timestamp_readback.Get(), 0);

			// Readback: every row's SeqState, and each (layer, kv head, K|V) run of this chunk's
			// committed positions -- contiguous in the K/V layout, so one copy per run.
			D3D12_RESOURCE_BARRIER pre_copy[2]{};
			pre_copy[0] = to_uav;
			pre_copy[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
			pre_copy[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
			pre_copy[1] = pre_copy[0];
			pre_copy[1].Transition.pResource = state.kv_uav.Get();
			dev.list->ResourceBarrier(2, pre_copy);
			if (external_kv_resident != nullptr && io_external_kv_needs_resume_barrier != nullptr) {
				*io_external_kv_needs_resume_barrier = true;
			}
			dev.list->CopyBufferRegion(g_t2790.seq_readback.Get(), 0, g_t2790.seq_rows.Get(), 0,
			                           g_t2790.seq_stride * kT2790Rows);
			const uint64_t run_bytes = static_cast<uint64_t>(n_real) * HD;
			const uint64_t kv_bytes = static_cast<uint64_t>(N) * NH * 2u * run_bytes;
			if (kv_bytes > g_t2790.kv_readback_bytes) {
				const uint64_t cap_bytes = static_cast<uint64_t>(N) * NH * 2u * kT2790Rows * HD;
				g_t2790.kv_readback = dev.MakeBuffer(cap_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
				                                     D3D12_RESOURCE_STATE_COPY_DEST);
				g_t2790.kv_readback_bytes = cap_bytes;
			}
			const uint64_t k_store = static_cast<uint64_t>(cap) * NH * HD;
			uint64_t dst = 0;
			for (uint32_t l = 0; l < N; ++l) {
				const uint64_t half_off = static_cast<uint64_t>(l) * k_store * 2u;
				for (uint32_t h = 0; h < NH; ++h) {
					const uint64_t row_off = static_cast<uint64_t>(h) * cap * HD + static_cast<uint64_t>(position0) * HD;
					dev.list->CopyBufferRegion(g_t2790.kv_readback.Get(), dst, state.kv_uav.Get(), half_off + row_off, run_bytes);
					dst += run_bytes;
					dev.list->CopyBufferRegion(g_t2790.kv_readback.Get(), dst, state.kv_uav.Get(),
					                           half_off + k_store + row_off, run_bytes);
					dst += run_bytes;
				}
			}
			D3D12_RESOURCE_BARRIER back_to_dest = to_uav;
			back_to_dest.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
			back_to_dest.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
			dev.list->ResourceBarrier(1, &back_to_dest);

			SSLM_GPU_HR(dev.list->Close());
			const auto t_record_end = std::chrono::steady_clock::now();
			ID3D12CommandList* lists[] = {dev.list.Get()};
			dev.queue->ExecuteCommandLists(1, lists);
			SSLM_GPU_HR(dev.queue->Signal(dev.fence.Get(), ++dev.fence_val));
			if (dev.fence->GetCompletedValue() < dev.fence_val) {
				SSLM_GPU_HR(dev.fence->SetEventOnCompletion(dev.fence_val, dev.fence_event));
				WaitForSingleObject(dev.fence_event, INFINITE);
			}
			const auto t_wait_end = std::chrono::steady_clock::now();
			g_t2790_record_ms += std::chrono::duration<double, std::milli>(t_record_end - t_record_start).count();
			g_t2790_wait_ms += std::chrono::duration<double, std::milli>(t_wait_end - t_record_end).count();

			if (dev.timestamp_frequency > 0 && n_stamps > 0) {
				std::vector<UINT64> ts(n_stamps + 1, 0);
				void* qp = nullptr;
				D3D12_RANGE qr{0, ts.size() * sizeof(UINT64)};
				SSLM_GPU_HR(dev.timestamp_readback->Map(0, &qr, &qp));
				std::memcpy(ts.data(), qp, ts.size() * sizeof(UINT64));
				D3D12_RANGE none{0, 0};
				dev.timestamp_readback->Unmap(0, &none);
				const double freq = static_cast<double>(dev.timestamp_frequency);
				for (uint32_t d = 0; d < n_stamps; ++d) {
					g_t2790_site_ms[dispatch_site[d]] += static_cast<double>(ts[d + 1] - ts[d]) / freq * 1000.0;
				}
				g_t2790_gpu_busy_ms += static_cast<double>(ts.back() - ts.front()) / freq * 1000.0;
			}
			++g_t2790_chunks;

			// Host update from the rows.
			std::vector<uint8_t> rows(static_cast<size_t>(g_t2790.seq_stride) * kT2790Rows);
			{
				void* p = nullptr;
				D3D12_RANGE range{0, rows.size()};
				SSLM_GPU_HR(g_t2790.seq_readback->Map(0, &range, &p));
				std::memcpy(rows.data(), p, rows.size());
				D3D12_RANGE none{0, 0};
				g_t2790.seq_readback->Unmap(0, &none);
			}
			auto row_u32 = [&](uint32_t r, uint32_t off) {
				uint32_t v;
				std::memcpy(&v, rows.data() + static_cast<size_t>(g_t2790.seq_stride) * r + off, 4);
				return v;
			};
			auto row_i64 = [&](uint32_t r, uint32_t off) {
				int64_t v;
				std::memcpy(&v, rows.data() + static_cast<size_t>(g_t2790.seq_stride) * r + off, 8);
				return v;
			};
			auto pair64 = [&](uint32_t r, uint32_t lo_off, uint32_t hi_off) {
				return (static_cast<uint64_t>(row_u32(r, hi_off)) << 32) | row_u32(r, lo_off);
			};
			for (uint32_t r = 0; r < n_real; ++r) {
				const int64_t tag = row_i64(r, SeqStickyOff(H));
				if (tag != 0) return DecodeStickyTag(tag);  // no replay path in the spike
			}
			for (uint32_t r = 0; r < n_real; ++r) {
				seq.kv_saturation_count += pair64(r, SeqSatLoOff(H), SeqSatHiOff(H));
				seq.kv_landing_saturation_count += pair64(r, SeqKvLandingSatLoOff(H), SeqKvLandingSatHiOff(H));
				seq.k_channel_landing_saturation_count +=
				    pair64(r, SeqKChannelLandingSatLoOff(H), SeqKChannelLandingSatHiOff(H));
				seq.rope_q_saturation_count += pair64(r, SeqRopeQSatLoOff(H), SeqRopeQSatHiOff(H));
				seq.rope_k_saturation_count += pair64(r, SeqRopeKSatLoOff(H), SeqRopeKSatHiOff(H));
			}
			const uint32_t last = n_real - 1;
			for (uint32_t i = 0; i < H; ++i) {
				int32_t v;
				std::memcpy(&v, rows.data() + static_cast<size_t>(g_t2790.seq_stride) * last + i * 4u, 4);
				seq.hidden_codes[i] = static_cast<int8_t>(v);
			}
			seq.hidden_scale.m = row_i64(last, SeqScaleOff(H) + 0);
			seq.hidden_scale.e = row_i64(last, SeqScaleOff(H) + 8);
			seq.layer_index = row_u32(last, SeqLayerIdxOff(H));
			seq.context_length = row_i64(last, SeqCtxLenOff(H));
			{
				void* p = nullptr;
				D3D12_RANGE range{0, static_cast<SIZE_T>(kv_bytes)};
				SSLM_GPU_HR(g_t2790.kv_readback->Map(0, &range, &p));
				const uint8_t* src = static_cast<const uint8_t*>(p);
				uint64_t s = 0;
				for (uint32_t l = 0; l < N; ++l) {
					const uint64_t half_off = static_cast<uint64_t>(l) * k_store * 2u;
					for (uint32_t h = 0; h < NH; ++h) {
						const uint64_t row_off = static_cast<uint64_t>(h) * cap * HD + static_cast<uint64_t>(position0) * HD;
						std::memcpy(workspace + half_off + row_off, src + s, run_bytes);
						s += run_bytes;
						std::memcpy(workspace + half_off + k_store + row_off, src + s, run_bytes);
						s += run_bytes;
					}
				}
				D3D12_RANGE none{0, 0};
				g_t2790.kv_readback->Unmap(0, &none);
			}
		} catch (const std::exception& e) {
			std::fprintf(stderr, "t2790 widened chunk: %s\n", e.what());
			InvalidateResidencyCachesOnThrow();
			dev.list->Close();
			const HRESULT removed = dev.dev->GetDeviceRemovedReason();
			return removed != S_OK ? superslm::SslmForwardStatus::GpuDeviceRemoved
			                       : superslm::SslmForwardStatus::GpuAllocationFailed;
		}
		done += n_real;
		if (done < chunk_len) seq.layer_index = 0;
	}
	return superslm::SslmForwardStatus::Ok;
}

#endif  // SUPERSLM_T2790_WIDE_H
