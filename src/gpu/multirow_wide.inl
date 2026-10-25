// Multi-row GPU plan (rev 1.3.2), slice MR1: the host side of the row-widened kernels, test-only. Included at
// the end of superslm_gpu.cpp, inside namespace superslm_gpu, only when SUPERSLM_MULTIROW_WIDE is defined; no
// shipped target defines it. Ported from the T-2790 spike's widened prefill host (its root signature, row
// buffers and per-site recording), with the three additions plan §3.1 makes to the spike: paged K/V (t10 and
// b1 bound as the one-row body binds them), adapter tails (the same 28 constants, per row), and per-row
// causal width (the shaders' own `g_width + g_row`; the host's widest row only sizes the attention groups).
// The spike's single global row set (E14) is not carried: each submission allocates its own (§3.4 makes the
// row buffers the chunk's; the chunk itself is MR2's).
//
// Scope, as MR1 needs it: one synchronous submission of a layer slice for one sequence's rows, read back for
// the one-layer smoke (multirow_wide.h, gpu_1p0.cpp). No recorder, no chunk state, no close, no snapshot:
// those are MR2's. These kernels are not frozen: plan §1 and §9 hold the freeze until the RTX 50-series root
// cause is folded into §2 and §3.
//
// Shape (plan §3.1 and §3.2). Each of a layer's sites is recorded once for all rows:
//   - the 19 non-GEMM sites run shaders_wide/<site>_wide.hlsl compiled at the variant (<site>_wide_h<H>.cso),
//     dispatched (groups_x, H, 1): group row y is sequence row y, with its own SeqState, LayerScratch and
//     WorkScratch views (root parameter kWideRowTableParam: one descriptor table, spaces 1 to 3) and its own
//     position (row 0's + y);
//   - the six GEMMs run proj_gemm_wide.hlsl (<g>_proj_gemm_wide_h<H>.cso), one pass over each weight row
//     serving every row, dispatched with the groups and the per-group thread count the one-row body takes
//     from ComputeGpuGemmSiteGroupPlan, unchanged (§3.2's wave-size rule).
// Every row shares the one K/V buffer, page table and page constants. The root signature is the composed one
// (harness::Device::FillComposedRootParams, parameters 0 to 16, identical) plus that table, so every
// binding the one-row body reads is bound here at the same parameter, from what
// PrepareGpuLayerLoopChunkOpenState bound.
#if defined(SUPERSLM_MULTIROW_WIDE)

namespace {

// The largest variant (plan §3.1: H in {2, 4, 8}); the descriptor table has this many views per space and a
// smaller variant's shaders declare a prefix of them.
constexpr uint32_t kWideMaxRows = 8;
constexpr UINT kWideRowTableParam = harness::Device::kComposedRootParamCount;  // the parameter after b1
constexpr UINT kWideRootParamCount = harness::Device::kComposedRootParamCount + 1;
// A padded row's sticky word: any nonzero tag every site returns on before touching anything (plan §3.1).
constexpr int64_t kWidePaddedRowTag = 99;

uint64_t WideAlign256(uint64_t x) { return (x + 255u) & ~uint64_t{255}; }

bool WideVariantSupported(uint32_t h) { return h == 2 || h == 4 || h == 8; }

// The composed signature's parameters with the row table appended: 60 + 1 = 61 of D3D12's 64 DWORDs.
Microsoft::WRL::ComPtr<ID3D12RootSignature> MakeWideRootSig(harness::Device& dev) {
	static_assert(harness::Device::kComposedRootSignatureDwords + 1 <= D3D12_MAX_ROOT_COST,
	              "the widened signature adds one descriptor-table DWORD to the composed one");
	D3D12_ROOT_PARAMETER composed[harness::Device::kComposedRootParamCount]{};
	harness::Device::FillComposedRootParams(composed);
	D3D12_ROOT_PARAMETER ps[kWideRootParamCount]{};
	for (UINT i = 0; i < harness::Device::kComposedRootParamCount; ++i) ps[i] = composed[i];
	D3D12_DESCRIPTOR_RANGE ranges[3]{};
	for (uint32_t i = 0; i < 3; ++i) {
		ranges[i].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		ranges[i].NumDescriptors = kWideMaxRows;
		ranges[i].BaseShaderRegister = 0;
		ranges[i].RegisterSpace = i + 1;  // space1 SeqState, space2 LayerScratch, space3 WorkScratch
		ranges[i].OffsetInDescriptorsFromTableStart = i * kWideMaxRows;
	}
	ps[kWideRowTableParam].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	ps[kWideRowTableParam].DescriptorTable.NumDescriptorRanges = 3;
	ps[kWideRowTableParam].DescriptorTable.pDescriptorRanges = ranges;
	ps[kWideRowTableParam].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	D3D12_ROOT_SIGNATURE_DESC rs{};
	rs.NumParameters = kWideRootParamCount;
	rs.pParameters = ps;
	Microsoft::WRL::ComPtr<ID3DBlob> blob, err;
	if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) {
		if (err) std::fprintf(stderr, "%s\n", static_cast<const char*>(err->GetBufferPointer()));
		throw std::runtime_error("multi-row widened root signature serialization failed");
	}
	Microsoft::WRL::ComPtr<ID3D12RootSignature> r;
	SSLM_GPU_HR(dev.dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
	                                          IID_PPV_ARGS(&r)));
	return r;
}

// A layer's sites in the one-row body's recording order (RecordOneTokenFullDepthDispatchBody).
enum WideSite : uint32_t {
	kWAttnNorm, kWQProjGemm, kWQProjTail, kWKvProjGemm, kWKvProjTail, kWQkNorm, kWRopeGuard, kWRopeCommit,
	kWAttentionScore, kWSoftmax, kWContextAccumulate, kWCtxFold, kWOProjGemm, kWOProjTail, kWAttnResidual,
	kWMlpNorm, kWGateProjGemm, kWGateProjTail, kWUpProjGemm, kWUpProjTail, kWMlpAct, kWDownProjGemm,
	kWDownProjTail, kWMlpResidual, kWCommit, kWideSiteCount
};
// Each site's widened shader base name; the variant's .cso is <name>_h<H>.cso (CMakeLists.txt, build.bat).
const char* const kWideSiteShader[kWideSiteCount] = {
    "attn_norm_site_wide",        "q_proj_gemm_wide",   "q_proj_site_wide",    "kv_proj_gemm_wide",
    "kv_proj_site_wide",          "qk_norm_site_wide",  "rope_guard_site_wide", "rope_commit_site_wide",
    "attention_score_site_wide",  "softmax_site_wide",  "context_accumulate_site_wide", "ctx_fold_site_wide",
    "o_proj_gemm_wide",           "o_proj_site_wide",   "attn_residual_site_wide", "mlp_norm_site_wide",
    "gate_proj_gemm_wide",        "gate_proj_site_wide", "up_proj_gemm_wide",  "up_proj_site_wide",
    "mlp_act_site_wide",          "down_proj_gemm_wide", "down_proj_site_wide", "mlp_residual_site_wide",
    "commit_site_wide"};

// The widened pipelines of one variant, sharing one root signature across every variant (the table is sized
// for the largest). Built on first use, as GetOrBuildComposedPipeline builds the one-row ones.
struct WidePipelines {
	Microsoft::WRL::ComPtr<ID3D12PipelineState> pso[kWideSiteCount];
};
Microsoft::WRL::ComPtr<ID3D12RootSignature>& WideRootSig() {
	static Microsoft::WRL::ComPtr<ID3D12RootSignature> s;
	return s;
}
WidePipelines& GetOrBuildWidePipelines(harness::Device& dev, uint32_t variant_h, bool has_qk_norm) {
	static WidePipelines by_variant[kWideMaxRows + 1];
	if (!WideRootSig()) WideRootSig() = MakeWideRootSig(dev);
	WidePipelines& p = by_variant[variant_h];
	for (uint32_t s = 0; s < kWideSiteCount; ++s) {
		if (p.pso[s] || (s == kWQkNorm && !has_qk_norm)) continue;
		const std::string name = std::string(kWideSiteShader[s]) + "_h" + std::to_string(variant_h);
		p.pso[s] = dev.MakePSO(WideRootSig().Get(), harness::ReadFile(harness::ShaderPath(name)));
	}
	return p;
}

// One submission's row buffers: `variant_h` rows of each per-row buffer, each row a 256-byte-aligned window
// of one buffer, and the shader-visible table of their views.
struct WideRowSet {
	Microsoft::WRL::ComPtr<ID3D12Resource> seq_rows, scratch_rows, work_rows, seq_readback, kv_readback;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
	uint64_t seq_stride = 0, scratch_stride = 0, work_stride = 0;
};

void MakeWideRowSet(harness::Device& dev, uint32_t variant_h, uint64_t seq_bytes, uint64_t scratch_bytes,
                    uint64_t work_bytes, uint64_t kv_bytes, WideRowSet* rs) {
	rs->seq_stride = WideAlign256(seq_bytes);
	rs->scratch_stride = WideAlign256(scratch_bytes);
	rs->work_stride = WideAlign256(work_bytes);
	rs->seq_rows = dev.MakeBuffer(rs->seq_stride * variant_h, D3D12_HEAP_TYPE_DEFAULT,
	                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
	rs->scratch_rows = dev.MakeBuffer(rs->scratch_stride * variant_h, D3D12_HEAP_TYPE_DEFAULT,
	                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
	                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	rs->work_rows = dev.MakeBuffer(rs->work_stride * variant_h, D3D12_HEAP_TYPE_DEFAULT,
	                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	rs->seq_readback = dev.MakeBuffer(rs->seq_stride * variant_h, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
	                                  D3D12_RESOURCE_STATE_COPY_DEST);
	rs->kv_readback = dev.MakeBuffer(kv_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
	                                 D3D12_RESOURCE_STATE_COPY_DEST);
	harness::NameBuffer(rs->seq_rows.Get(), L"SuperSLM SeqState rows (widened)");
	harness::NameBuffer(rs->scratch_rows.Get(), L"SuperSLM LayerScratch rows (widened)");
	harness::NameBuffer(rs->work_rows.Get(), L"SuperSLM WorkScratch rows (widened)");
	D3D12_DESCRIPTOR_HEAP_DESC hd{};
	hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	hd.NumDescriptors = 3 * kWideMaxRows;
	hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	SSLM_GPU_HR(dev.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rs->heap)));
	const UINT inc = dev.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	const D3D12_CPU_DESCRIPTOR_HANDLE base = rs->heap->GetCPUDescriptorHandleForHeapStart();
	ID3D12Resource* const bufs[3] = {rs->seq_rows.Get(), rs->scratch_rows.Get(), rs->work_rows.Get()};
	const uint64_t strides[3] = {rs->seq_stride, rs->scratch_stride, rs->work_stride};
	for (uint32_t k = 0; k < 3; ++k) {
		for (uint32_t r = 0; r < kWideMaxRows; ++r) {
			// Views past the variant are never indexed (the variant's shaders declare SSLM_WIDE_H of them);
			// they repeat row 0's so every descriptor in the table is a valid one.
			const uint32_t row = r < variant_h ? r : 0u;
			D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
			ud.Format = DXGI_FORMAT_R32_TYPELESS;
			ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			ud.Buffer.FirstElement = (strides[k] * row) / 4u;
			ud.Buffer.NumElements = static_cast<UINT>(strides[k] / 4u);
			ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
			D3D12_CPU_DESCRIPTOR_HANDLE h = base;
			h.ptr += static_cast<SIZE_T>(k * kWideMaxRows + r) * inc;
			dev.dev->CreateUnorderedAccessView(bufs[k], nullptr, &ud, h);
		}
	}
}

// The widened body for one layer slice: the one-row body's sites, constants and order, once for all rows.
// `plan_for` is the one-row body's six GEMM plans, from ComputeGpuGemmSiteGroupPlan.
void RecordWideLayerSliceBody(harness::Device& dev, const WidePipelines& pipes, uint32_t variant_h,
                              uint32_t start_layer, uint32_t layers, const GpuLayerLoopChunkOpenState& state,
                              uint32_t position0, const GpuAdapterBridge* adapter_bridge, bool has_qk_norm) {
	const uint32_t H = state.H, HD = state.HD, NH = state.NH, NQH = state.NQH, I = state.I, N = state.N;
	const uint32_t cap = state.context_cap_u32;
	const uint32_t Q_WIDTH = NQH * HD;
	// Row 0's causal width; row r's is width0 + r, applied in the shaders (plan §3.1).
	const uint32_t width0 = position0 + 1u;
	D3D12_RESOURCE_BARRIER uav_barrier{};
	uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	uav_barrier.UAV.pResource = nullptr;  // global, as every barrier the one-row body issues

	// A GEMM site: the plan's groups in X, one Y row (the shader loops over the rows), and the plan's
	// per-group thread count as the 11th constant, exactly the one-row bind_and_dispatch's.
	auto gemm = [&](uint32_t site, uint32_t layer, const GpuGemmSiteGroupPlan& plan) {
		const uint32_t consts[12] = {layer, H, HD, NH, cap, position0, NQH, width0, I, N, plan.lanes, Q_WIDTH};
		dev.list->SetComputeRoot32BitConstants(0, 12, consts, 0);
		dev.list->SetPipelineState(pipes.pso[site].Get());
		harness::RecordDispatch(dev.list.Get(), plan.groups, 1, 1);
		harness::RecordBarriers(dev.list.Get(), 1, &uav_barrier);
	};
	// A plain site: the one-row bind_and_dispatch's 12 values with no GEMM plan (its 11th value is 1), rows
	// in Y.
	auto plain = [&](uint32_t site, uint32_t layer, uint32_t groups_x) {
		const uint32_t consts[12] = {layer, H, HD, NH, cap, position0, NQH, width0, I, N, 1u, Q_WIDTH};
		dev.list->SetComputeRoot32BitConstants(0, 12, consts, 0);
		dev.list->SetPipelineState(pipes.pso[site].Get());
		harness::RecordDispatch(dev.list.Get(), groups_x, variant_h, 1);
		harness::RecordBarriers(dev.list.Get(), 1, &uav_barrier);
	};
	// A tail site: the one-row bind_and_dispatch_tail's 28 values, the adapter slots filled from the one
	// bridge exactly as it fills them (one adapter per chunk, plan §3.1/§4.4), rows in Y.
	struct Slot {
		int slot_p;
		uint32_t in_base;
		uint32_t wide_base;
	};
	auto tail = [&](uint32_t site, uint32_t layer, std::initializer_list<Slot> slots) {
		uint32_t consts[28] = {layer, H, HD, NH, cap, position0, NQH, width0, I, N, 1u};
		size_t base = 11;
		for (const Slot& s : slots) {
			uint32_t rank = 0, a_off = 0, b_off = 0, fold_off = 0;
			if (adapter_bridge) {
				const AdapterProjSlot& slot =
				    adapter_bridge->slots[static_cast<size_t>(layer) * 7u + static_cast<size_t>(s.slot_p)];
				if (slot.present) {
					rank = slot.rank;
					a_off = static_cast<uint32_t>(slot.a_offset);
					b_off = static_cast<uint32_t>(slot.b_offset);
					fold_off = static_cast<uint32_t>(slot.fold_offset);
				}
			}
			consts[base + 0] = rank;
			consts[base + 1] = a_off;
			consts[base + 2] = b_off;
			consts[base + 3] = fold_off;
			consts[base + 4] = static_cast<uint32_t>(state.work_adapter_u_off);
			consts[base + 5] = s.in_base;
			consts[base + 6] = s.wide_base;
			consts[base + 7] = Stage1LanesForRank(rank);
			base += 8;
		}
		consts[27] = Q_WIDTH;
		dev.list->SetComputeRoot32BitConstants(0, 28, consts, 0);
		dev.list->SetPipelineState(pipes.pso[site].Get());
		harness::RecordDispatch(dev.list.Get(), 1, variant_h, 1);
		harness::RecordBarriers(dev.list.Get(), 1, &uav_barrier);
	};

	const uint32_t KV2 = 2u * NH * HD;
	const GpuGemmSiteGroupPlan q_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::QProj, H, KV2, I, Q_WIDTH);
	const GpuGemmSiteGroupPlan o_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::OProj, H, KV2, I);
	const GpuGemmSiteGroupPlan kv_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::KvProj, H, KV2, I);
	const GpuGemmSiteGroupPlan gate_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::GateProj, H, KV2, I);
	const GpuGemmSiteGroupPlan up_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::UpProj, H, KV2, I);
	const GpuGemmSiteGroupPlan down_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::DownProj, H, KV2, I);
	for (const GpuGemmSiteGroupPlan* plan : {&q_plan, &o_plan, &kv_plan, &gate_plan, &up_plan, &down_plan}) {
		if (static_cast<uint64_t>(plan->groups) * static_cast<uint64_t>(plan->channels_per_group) <
		    static_cast<uint64_t>(plan->out_channels)) {
			throw GpuGemmGroupArithmeticError(
			    "RecordWideLayerSliceBody: GEMM group plan does not cover its own out_channels");
		}
	}
	// Attention is sized by the widest row (row variant_h - 1); narrower rows' surplus threads return at the
	// shader's own item bound. Only the group count uses it (plan §3.1, MX1).
	const uint32_t max_width = width0 + (variant_h - 1u);
	const uint32_t attn_score_groups = (NQH * max_width + 255u) / 256u;
	const uint32_t ctx_accum_groups = (NQH * HD + 255u) / 256u;
	const uint32_t rope_groups = (NQH * (HD / 2u) + 255u) / 256u;
	const GpuScratchLayout& sl = state.scratch_layout;
	const uint32_t wa = static_cast<uint32_t>(state.work_wide_a_off);
	const uint32_t wb = static_cast<uint32_t>(state.work_wide_b_off);

	for (uint32_t i = 0; i < layers; ++i) {
		const uint32_t l = start_layer + i;
		plain(kWAttnNorm, l, 1);
		gemm(kWQProjGemm, l, q_plan);
		tail(kWQProjTail, l, {{/*q=*/0, sl.normed, wa}});
		gemm(kWKvProjGemm, l, kv_plan);
		tail(kWKvProjTail, l, {{/*k=*/5, sl.normed, wa}, {/*v=*/6, sl.normed, wb}});
		if (has_qk_norm) plain(kWQkNorm, l, NQH + NH);
		plain(kWRopeGuard, l, rope_groups);
		plain(kWRopeCommit, l, rope_groups);
		plain(kWAttentionScore, l, attn_score_groups);
		plain(kWSoftmax, l, 1);
		plain(kWContextAccumulate, l, ctx_accum_groups);
		plain(kWCtxFold, l, 1);
		gemm(kWOProjGemm, l, o_plan);
		tail(kWOProjTail, l, {{/*o=*/1, sl.ctx_codes, wa}});
		plain(kWAttnResidual, l, 1);
		plain(kWMlpNorm, l, 1);
		gemm(kWGateProjGemm, l, gate_plan);
		tail(kWGateProjTail, l, {{/*gate=*/2, sl.normed, wa}});
		gemm(kWUpProjGemm, l, up_plan);
		tail(kWUpProjTail, l, {{/*up=*/3, sl.normed, wa}});
		plain(kWMlpAct, l, 1);
		gemm(kWDownProjGemm, l, down_plan);
		tail(kWDownProjTail, l, {{/*down=*/4, sl.act_codes, wa}});
		plain(kWMlpResidual, l, 1);
		plain(kWCommit, l, 1);
	}
}

}  // namespace

superslm::SslmForwardStatus RunMultirowWideLayersForSmoke(
    const superslm::SequenceLayerState* rows, uint32_t real_rows, uint32_t variant_h, uint32_t layers,
    uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
    size_t intermediate_size, int64_t context_cap, uint8_t* workspace, size_t workspace_size,
    ID3D12Resource* external_kv_resident, bool* io_external_kv_needs_resume_barrier,
    ID3D12Resource* external_weights_resident, ID3D12Resource* external_rope_cos_resident,
    ID3D12Resource* external_rope_sin_resident, bool external_rope_has, uint64_t external_rope_cos_elems,
    uint64_t external_rope_sin_elems, const GpuAdapterBridge* adapter_bridge, size_t q_width,
    uint64_t model_generation, bool model_has_qk_norm, const GpuPagedKvBinding* paged_kv,
    GpuWideRowsReadback* out) {
	if (rows == nullptr || out == nullptr || external_kv_resident == nullptr || !WideVariantSupported(variant_h) ||
	    real_rows == 0 || real_rows > variant_h || layers == 0) {
		return superslm::SslmForwardStatus::InvalidLayerBudget;  // a malformed test call; nothing recorded
	}
	static const superslm::SslmTensorManifest kEmptyManifest;
	harness::Device& dev = harness::GetDevice();
	const superslm::KvPageView kv_view =
	    paged_kv != nullptr ? paged_kv->view : GpuOnePageView(workspace, workspace_size, context_cap);
	// Prepare runs on a copy of row 0 (it reads the state for its guards and packs the one-row SeqState it
	// binds at u0, which no widened shader declares).
	superslm::SequenceLayerState row0 = rows[0];
	GpuLayerLoopChunkOpenState state;
	WideRowSet rs;
	Microsoft::WRL::ComPtr<ID3D12Resource> init_upload;
	const bool kv_latch_entry = io_external_kv_needs_resume_barrier != nullptr && *io_external_kv_needs_resume_barrier;
	bool executed = false;
	try {
		const superslm::SslmForwardStatus prep = PrepareGpuLayerLoopChunkOpenState(
		    dev, row0, /*layers=*/nullptr, num_hidden_layers, layers, hidden_size, head_dim, num_key_value_heads,
		    intermediate_size, context_cap, kEmptyManifest, kv_view, workspace_size,
		    paged_kv != nullptr ? paged_kv->table_resident : nullptr, paged_kv != nullptr && paged_kv->upload_table,
		    /*coverage_tokens=*/real_rows, external_kv_resident, io_external_kv_needs_resume_barrier,
		    external_weights_resident, external_rope_cos_resident, external_rope_sin_resident, external_rope_has,
		    external_rope_cos_elems, external_rope_sin_elems, adapter_bridge, &state, q_width, model_generation);
		if (prep != superslm::SslmForwardStatus::Ok) return prep;
		const uint32_t H = state.H;
		const uint64_t kv_bytes = external_kv_resident->GetDesc().Width;
		const WidePipelines& pipes = GetOrBuildWidePipelines(dev, variant_h, model_has_qk_norm);
		MakeWideRowSet(dev, variant_h, SeqTotalSize(H), state.scratch_uav->GetDesc().Width,
		               state.work_scratch_uav->GetDesc().Width, kv_bytes, &rs);

		// Each row's SeqState as a one-row submission packs it; padded rows repeat the last real row's with
		// their own position and a set sticky word, so every site returns on them first (plan §3.1).
		const int64_t p0 = rows[0].context_length;
		std::vector<uint8_t> init(static_cast<size_t>(rs.seq_stride) * variant_h, 0);
		for (uint32_t r = 0; r < variant_h; ++r) {
			std::vector<uint8_t> row = PackGpuSeqStateBytes(rows[r < real_rows ? r : real_rows - 1], H);
			if (r >= real_rows) {
				PutI64At(row, SeqCtxLenOff(H), p0 + static_cast<int64_t>(r));
				PutI64At(row, SeqStickyOff(H), kWidePaddedRowTag);
			}
			std::memcpy(init.data() + static_cast<size_t>(rs.seq_stride) * r, row.data(), row.size());
		}
		init_upload = dev.Upload(init.data(), init.size());
		harness::RecordCopyBufferRegion(dev.list.Get(), rs.seq_rows.Get(), 0, init_upload.Get(), 0, init.size());
		harness::RecordTransition(dev.list.Get(), rs.seq_rows.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
		                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

		// Bindings: the composed parameters, from what Prepare bound, then the row table.
		dev.list->SetComputeRootSignature(WideRootSig().Get());
		ID3D12DescriptorHeap* heaps[] = {rs.heap.Get()};
		dev.list->SetDescriptorHeaps(1, heaps);
		harness::BindRootSrv(dev.list.Get(), 1, state.bound_weights.Get());
		harness::BindRootSrv(dev.list.Get(), 2, state.layout_buf.Get());
		harness::BindRootSrv(dev.list.Get(), 3, state.rope_buf.Get());
		harness::BindRootSrv(dev.list.Get(), 4, state.model_const_buf.Get());
		harness::BindRootSrv(dev.list.Get(), 5, state.silu_lut_buf.Get());
		harness::BindRootSrv(dev.list.Get(), 6, state.bound_rope_cos.Get());
		harness::BindRootSrv(dev.list.Get(), 7, state.bound_rope_sin.Get());
		harness::BindRootSrv(dev.list.Get(), 8, state.scratch_layout_buf.Get());
		// u0, u1 and u3 (space 0) are declared by the composed parameters and read by no widened shader;
		// bound to the one-row buffers so that every root descriptor is a valid address.
		harness::BindRootUav(dev.list.Get(), 9, state.seq_uav.Get());
		harness::BindRootUav(dev.list.Get(), 10, state.scratch_uav.Get());
		harness::BindRootUav(dev.list.Get(), 11, state.kv_uav.Get());
		harness::BindRootUav(dev.list.Get(), 12, state.work_scratch_uav.Get());
		harness::BindRootSrv(dev.list.Get(), 13, state.bound_lora_ab.Get());
		harness::BindRootSrv(dev.list.Get(), 14, state.bound_fold.Get());
		harness::BindRootSrv(dev.list.Get(), harness::Device::kComposedRootParamKvPageTable,
		                     state.bound_kv_page_table.Get());
		dev.list->SetComputeRoot32BitConstants(harness::Device::kComposedRootParamKvPageConstants,
		                                       harness::Device::kComposedPageConstantCount,
		                                       state.bound_kv_page_constants, 0);
		dev.list->SetComputeRootDescriptorTable(kWideRowTableParam, rs.heap->GetGPUDescriptorHandleForHeapStart());

		RecordWideLayerSliceBody(dev, pipes, variant_h, rows[0].layer_index, layers, state,
		                         static_cast<uint32_t>(p0), adapter_bridge, model_has_qk_norm);

		// Readback: every row's SeqState and the whole K/V buffer. The K/V buffer rests in COPY_SOURCE
		// afterwards, as after every one-row readback, so the latch says so.
		D3D12_RESOURCE_BARRIER pre_copy[3]{};
		uint32_t n_pre = 0;
		for (ID3D12Resource* r : {rs.seq_rows.Get(), state.kv_uav.Get()}) {
			pre_copy[n_pre].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			pre_copy[n_pre].Transition.pResource = r;
			pre_copy[n_pre].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
			pre_copy[n_pre].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
			pre_copy[n_pre].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
			++n_pre;
		}
		n_pre += TableBackToCommonBarrier(state, &pre_copy[n_pre]);
		harness::RecordBarriers(dev.list.Get(), n_pre, pre_copy);
		if (io_external_kv_needs_resume_barrier != nullptr) *io_external_kv_needs_resume_barrier = true;
		harness::RecordCopyBufferRegion(dev.list.Get(), rs.seq_readback.Get(), 0, rs.seq_rows.Get(), 0,
		                                rs.seq_stride * variant_h);
		harness::RecordCopyBufferRegion(dev.list.Get(), rs.kv_readback.Get(), 0, state.kv_uav.Get(), 0, kv_bytes);

		dev.CloseListOrThrow();
		executed = true;
		dev.SubmitAndWaitKeepingAlive(true, init_upload, rs.seq_rows, rs.scratch_rows, rs.work_rows,
		                              rs.seq_readback, rs.kv_readback, state.seq_uav, state.scratch_uav,
		                              state.work_scratch_uav, state.layout_buf, state.rope_buf, state.model_const_buf,
		                              state.silu_lut_buf, state.scratch_layout_buf);

		out->row_bytes = SeqTotalSize(H);
		out->entry_rows.assign(static_cast<size_t>(out->row_bytes) * variant_h, 0);
		for (uint32_t r = 0; r < variant_h; ++r) {
			std::memcpy(out->entry_rows.data() + out->row_bytes * r,
			            init.data() + static_cast<size_t>(rs.seq_stride) * r, out->row_bytes);
		}
		out->rows.assign(static_cast<size_t>(out->row_bytes) * variant_h, 0);
		{
			void* p = nullptr;
			const D3D12_RANGE range{0, static_cast<SIZE_T>(rs.seq_stride * variant_h)};
			SSLM_GPU_HR(rs.seq_readback->Map(0, &range, &p));
			for (uint32_t r = 0; r < variant_h; ++r) {
				std::memcpy(out->rows.data() + out->row_bytes * r,
				            static_cast<const uint8_t*>(p) + static_cast<size_t>(rs.seq_stride) * r, out->row_bytes);
			}
			const D3D12_RANGE none{0, 0};
			rs.seq_readback->Unmap(0, &none);
		}
		out->kv.assign(static_cast<size_t>(kv_bytes), 0);
		{
			void* p = nullptr;
			const D3D12_RANGE range{0, static_cast<SIZE_T>(kv_bytes)};
			SSLM_GPU_HR(rs.kv_readback->Map(0, &range, &p));
			std::memcpy(out->kv.data(), p, out->kv.size());
			const D3D12_RANGE none{0, 0};
			rs.kv_readback->Unmap(0, &none);
		}
	} catch (...) {
		// The latch goes back to what this call found when nothing it recorded was submitted.
		const bool list_closed = executed || AbandonRecording(dev, state);
		if (!executed && io_external_kv_needs_resume_barrier != nullptr) {
			*io_external_kv_needs_resume_barrier = kv_latch_entry;
		}
		return ClassifyForwardFault(dev, !list_closed);
	}
	return superslm::SslmForwardStatus::Ok;
}

#endif  // SUPERSLM_MULTIROW_WIDE
