// Streamed LoRA adapters (plan: superslm-lora-residency rev 1.2, §3.1 and §3.3): the device-free
// adapter core shared by the CPU map (src/sslm_abi.cpp) and the GPU prepare (src/gpu/gpu_1p0.cpp).
//
//   - PrescreenAdapterArtifact: an allocation-free read of the container's section table that
//     refuses, before any byte is parsed, an artifact an adapter is never shaped like. It is what
//     makes the peak bounds below finite and tight: the parse of every section an adapter may
//     carry is bounded by the counts this check caps, and the sections it refuses (a tokenizer
//     above all, whose parse grows with its content) are never parsed on an adapter path.
//   - AdapterPeakBound*: the high-water mark of host heap bytes a CPU map or a GPU prepare can
//     allocate, from the base geometry and the artifact's byte size alone. Every term names the
//     allocation sites it covers ("PEAK-BOUND <term>" comments at those sites, and the census in
//     docs/lora-residency/peak-bound-census.md, which tools/ci/check_adapter_peak_bound.py holds to
//     the source).
//   - PrepareAdapterCore / PackAdapterSlots: today's GPU map packing (A, B int8 rows and the
//     12-byte fold rows, one slot per (layer, projection)), moved here unchanged so it runs on any
//     thread and builds on every platform. It reads no device and no model handle: only the
//     geometry snapshot it is given.
//
// Header-only and internal (src/, never installed), so every build list that compiles
// sslm_abi.cpp or gpu_1p0.cpp gets it with no new source file.
#ifndef SUPERSLM_SRC_ADAPTER_CORE_H
#define SUPERSLM_SRC_ADAPTER_CORE_H

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <vector>

#include "superslm/adapter_info.h"
#include "superslm/adapter_marshal.h"
#include "superslm/artifact.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"

namespace superslm_adapter {

// ---- the prescreen ------------------------------------------------------------------------------

// The six section types the converter writes (tools/sslm_convert_adapter.py, `sections = [...]`).
inline constexpr superslm::SslmSectionType kAdapterSectionTypes[] = {
    superslm::SslmSectionType::Config,          superslm::SslmSectionType::SigmoidLut,
    superslm::SslmSectionType::Provenance,      superslm::SslmSectionType::Weights,
    superslm::SslmSectionType::DeltaFoldScales, superslm::SslmSectionType::UFoldScales};
inline constexpr uint32_t kAdapterMaxSections = 6;
// The ADP1 name an adapter may carry. The converter writes the PEFT adapter's own name; the info
// struct keeps 63 bytes of it. Capping it keeps the name's copy (AdapterMeta) out of the bound's
// size-proportional terms.
inline constexpr uint32_t kAdapterMaxNameBytes = 4096;
// Per base layer: lora_A and lora_B for each of the seven projections, and one DFS1 and one UFS1
// entry for each.
inline constexpr uint64_t kAdapterProjections = 7;

inline uint32_t PrescreenRdU32(const uint8_t* p) noexcept {
	return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
inline uint64_t PrescreenRdU64(const uint8_t* p) noexcept {
	return uint64_t(PrescreenRdU32(p)) | (uint64_t(PrescreenRdU32(p + 4)) << 32);
}

inline uint64_t AdapterMaxManifestEntries(superslm::SslmSectionType type, uint32_t num_hidden_layers) noexcept {
	const uint64_t per_layer = type == superslm::SslmSectionType::Weights ? 2 * kAdapterProjections
	                                                                       : kAdapterProjections;
	return per_layer * num_hidden_layers;
}

// True when the artifact may go on to SslmModel::Load on an adapter path. False refuses it (the
// caller's artifact rejection) for: more than six sections; a section type outside the six; a
// Weights, DeltaFoldScales or UFoldScales manifest declaring more entries than an adapter for a
// base of `num_hidden_layers` layers can use; a Provenance section longer than ADP1's fixed prefix
// plus kAdapterMaxNameBytes. Reads only what the container's own bounds make readable and
// allocates nothing; anything it cannot read (a short header, a table or manifest header past the
// end) it leaves to SslmModel::Load, whose refusal of such input allocates only its fixed messages.
inline bool PrescreenAdapterArtifact(const uint8_t* data, size_t size, uint32_t num_hidden_layers) noexcept {
	if (!data || size < superslm::kHeaderBytes) return true;
	const uint32_t section_count = PrescreenRdU32(data + 12);
	if (section_count > kAdapterMaxSections) return false;
	const uint64_t table_end = uint64_t(superslm::kHeaderBytes) + uint64_t(section_count) * superslm::kSectionDescBytes;
	if (table_end > size) return true;
	for (uint32_t i = 0; i < section_count; ++i) {
		const uint8_t* row = data + superslm::kHeaderBytes + size_t(i) * superslm::kSectionDescBytes;
		const auto type = static_cast<superslm::SslmSectionType>(PrescreenRdU32(row + 0));
		const uint64_t offset = PrescreenRdU64(row + 8);
		const uint64_t byte_size = PrescreenRdU64(row + 16);
		bool allowed = false;
		for (superslm::SslmSectionType t : kAdapterSectionTypes) allowed = allowed || t == type;
		if (!allowed) return false;
		if (type == superslm::SslmSectionType::Provenance &&
		    byte_size > uint64_t(kAdp1FixedBytes) + kAdapterMaxNameBytes)
			return false;
		const bool manifest = type == superslm::SslmSectionType::Weights ||
		                      type == superslm::SslmSectionType::DeltaFoldScales ||
		                      type == superslm::SslmSectionType::UFoldScales;
		if (manifest && byte_size >= superslm::kManifestHeaderBytes && offset <= size &&
		    superslm::kManifestHeaderBytes <= size - offset) {
			const uint32_t tensor_count = PrescreenRdU32(data + offset + 8);
			if (tensor_count > AdapterMaxManifestEntries(type, num_hidden_layers)) return false;
		}
	}
	return true;
}

// ---- the peak bounds ----------------------------------------------------------------------------
//
// One term per allocation-site family on the map's and the prepare's call graph, after the
// prescreen (docs/lora-residency/peak-bound-census.md lists every site and its term). The sum is
// over-approximate by construction: every term is that family's worst case, and the families are
// added as though all were live at once.

// The artifact library's private Placed row (src/artifact.cpp): five fields, at most 48 bytes.
inline constexpr uint64_t kPlacedRowBytesMax = 48;
// Every rejection message the parse can build is a fixed text with numbers in it: no artifact
// byte (no name) reaches one once the adapter paths pass a null error string. A concatenation
// holds at most three such temporaries at once.
inline constexpr uint64_t kRejectionMessageBytesMax = 3 * 1024;
// The validators' per-layer probe names ("layer<N>.q_norm.gain", ValidateQkChannelTableAdditive),
// two at once; N is a u32, so each is under 32 bytes plus the allocator's terminator.
inline constexpr uint64_t kValidatorProbeBytesMax = 2 * 64;

// What std::allocator adds to a block on the target compiler. MSVC's allocator on x86 and x64 takes
// a block of 4096 bytes or more (its "big allocation") as `n + 47` bytes from ::operator new (55 in
// a debug build), to align it to 32 bytes; libstdc++ and libc++ add nothing. The info figures count
// every allocator block through AllocatorBlockBytes, and the peak terms add the overhead once for
// each block of their site that can reach the threshold.
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#if defined(_DEBUG)
inline constexpr uint64_t kAllocatorBigBlockOverhead = 55;
#else
inline constexpr uint64_t kAllocatorBigBlockOverhead = 47;
#endif
#else
inline constexpr uint64_t kAllocatorBigBlockOverhead = 0;
#endif
inline constexpr uint64_t kAllocatorBigBlockThreshold = 4096;
inline uint64_t AllocatorBlockBytes(uint64_t n) noexcept {
	return n >= kAllocatorBigBlockThreshold ? n + kAllocatorBigBlockOverhead : n;
}

struct AdapterPeakTerms {
	uint64_t artifact_copy = 0;   // PEAK-BOUND copy: SslmArtifact's owned bytes_ (OpenFromMemory)
	uint64_t section_rows = 0;    // PEAK-BOUND sections: Placed rows and the SslmSectionView table
	uint64_t manifests = 0;       // PEAK-BOUND manifests: tensor/fold tables, ranges, name sets
	uint64_t adapter_name = 0;    // PEAK-BOUND name: AdapterMeta::source_adapter_name
	uint64_t layer_table = 0;     // PEAK-BOUND layers: layer_adapters (one LayerAdapter per base layer)
	uint64_t claimed_modules = 0; // PEAK-BOUND modules: TargetModulesFromMask's vector, with growth
	uint64_t messages = 0;        // PEAK-BOUND messages: rejection and validator-probe strings
	uint64_t handle = 0;          // PEAK-BOUND handle: the CPU sslm_adapter_s / the prepared object
	uint64_t packed = 0;          // PEAK-BOUND packed: GPU prepare's ab_bytes + fold_bytes
	uint64_t slot_table = 0;      // PEAK-BOUND slots: GPU prepare's [layer][7] slot table

	uint64_t Sum() const noexcept {
		return artifact_copy + section_rows + manifests + adapter_name + layer_table + claimed_modules +
		       messages + handle + packed + slot_table;
	}
};

// The live bytes of one manifest parse of `n` entries of `entry_bytes` each, final table and the
// parse's transients together: the table (reserved exactly), the [start, end) ranges (reserved
// exactly), and FixedIntSet's slots (BucketCountFor(n) <= 4n slots of a string_view and a flag).
inline uint64_t ManifestParseBytesMax(uint64_t n, uint64_t entry_bytes) noexcept {
	constexpr uint64_t kRangeBytes = sizeof(std::pair<uint64_t, uint64_t>);
	constexpr uint64_t kSlotBytes = sizeof(std::string_view) + alignof(std::string_view);
	return n * entry_bytes + n * kRangeBytes + 4 * (n ? n : 1) * kSlotBytes;
}

inline AdapterPeakTerms AdapterPeakTermsFor(uint32_t num_hidden_layers, uint64_t artifact_bytes,
                                            uint64_t handle_bytes, bool gpu_prepare) noexcept {
	AdapterPeakTerms t;
	t.artifact_copy = artifact_bytes + kAllocatorBigBlockOverhead;
	t.section_rows = kAdapterMaxSections * (kPlacedRowBytesMax + sizeof(superslm::SslmSectionView));
	const uint64_t w = AdapterMaxManifestEntries(superslm::SslmSectionType::Weights, num_hidden_layers);
	const uint64_t f = AdapterMaxManifestEntries(superslm::SslmSectionType::DeltaFoldScales, num_hidden_layers);
	// Each of the three parses holds at most three allocator blocks at once (table, ranges, slots).
	t.manifests = ManifestParseBytesMax(w, sizeof(superslm::SslmTensorView)) +
	              2 * ManifestParseBytesMax(f, sizeof(superslm::SslmAmplifyingFoldEntry)) + 9 * kAllocatorBigBlockOverhead;
	// The longest name's buffer, its capacity rounded up to a multiple of 16 as MSVC's string does
	// (libstdc++ allocates exactly), plus the terminator.
	t.adapter_name = (kAdapterMaxNameBytes | 15) + 1 + kAllocatorBigBlockOverhead;
	t.layer_table = uint64_t(num_hidden_layers) * sizeof(superslm::LayerAdapter) + kAllocatorBigBlockOverhead;
	// Seven string_views pushed one at a time. libstdc++ doubles (1, 2, 4, 8: the last growth holds
	// 4 and 8 at once); MSVC grows by half (1, 2, 3, 4, 6, 9: the last holds 6 and 9). The larger.
	t.claimed_modules = (6 + 9) * sizeof(std::string_view);
	t.messages = kRejectionMessageBytesMax + kValidatorProbeBytesMax;
	t.handle = handle_bytes;
	if (gpu_prepare) {
		// Every packed byte is a copy of a lora_A, lora_B, DeltaFoldScales or UFoldScales row the
		// artifact holds, each in its own non-overlapping range (the manifests' overlap checks), so
		// the two buffers together never exceed the artifact; plus the one-byte floor each takes
		// when it would be empty.
		t.packed = artifact_bytes + 2 + 2 * kAllocatorBigBlockOverhead;
		t.slot_table = uint64_t(num_hidden_layers) * sizeof(std::array<superslm_gpu::AdapterProjSlot, 7>) + kAllocatorBigBlockOverhead;
	}
	return t;
}

// ---- info ---------------------------------------------------------------------------------------

inline uint64_t RoundUp64KiB(uint64_t n) noexcept { return (n + 65535u) / 65536u * 65536u; }

inline uint32_t CoveredLayers(const std::vector<superslm::LayerAdapter>& layers) noexcept {
	uint32_t covered = 0;
	for (const superslm::LayerAdapter& la : layers) {
		const superslm::LayerAdapterProjection* p[7] = {&la.q, &la.o, &la.gate, &la.up, &la.down, &la.k, &la.v};
		bool any = false;
		for (const auto* x : p) any = any || x->a_weight != nullptr;
		covered += any ? 1u : 0u;
	}
	return covered;
}

// The fields every info verb fills the same way. `info->struct_size` is the caller's and is kept.
inline void FillAdapterInfoCommon(const AdapterMeta& meta, uint32_t covered_layers,
                                  const std::array<uint8_t, superslm::kIntegrityHashBytes>& artifact_hash,
                                  sslm_adapter_info* info) noexcept {
	info->adp1_version = kAdp1Version;
	info->rank = meta.rank;
	info->target_modules_mask = meta.target_modules_mask;
	info->covered_layers = covered_layers;
	info->use_rslora = meta.use_rslora ? 1u : 0u;
	info->lora_alpha = meta.lora_alpha;
	std::memcpy(info->base_artifact_hash, meta.base_artifact_hash.data(), 32);
	std::memcpy(info->artifact_hash, artifact_hash.data(), 32);
	std::memset(info->name, 0, sizeof(info->name));
	const size_t n = meta.source_adapter_name.size() < sizeof(info->name) - 1 ? meta.source_adapter_name.size()
	                                                                         : sizeof(info->name) - 1;
	std::memcpy(info->name, meta.source_adapter_name.data(), n);
}

// The heap bytes a std::string owns: 0 while its characters sit in the object (the short-string
// buffer), else its capacity plus the terminator.
inline uint64_t StringHeapBytes(const std::string& s) noexcept {
	const char* d = s.data();
	const char* o = reinterpret_cast<const char*>(&s);
	if (d >= o && d < o + sizeof(std::string)) return 0;
	return AllocatorBlockBytes(uint64_t(s.capacity()) + 1);
}

// Every heap byte an adapter's parsed view and populated tables own (the CPU handle's, less the
// handle itself): the artifact copy (`artifact_bytes`, which OpenFromMemory checked equals the
// header's file size), the section table (reserved to the header's `section_count`), the three
// manifest tables, the per-layer table and the ADP1 name.
inline uint64_t AdapterViewHeapBytes(uint64_t artifact_bytes, uint32_t section_count,
                                     const superslm::SslmModelView& view, const AdapterMeta& meta,
                                     const std::vector<superslm::LayerAdapter>& layers) noexcept {
	return AllocatorBlockBytes(artifact_bytes) +
	       AllocatorBlockBytes(uint64_t(section_count) * sizeof(superslm::SslmSectionView)) +
	       AllocatorBlockBytes(view.weights.Tensors().capacity() * sizeof(superslm::SslmTensorView)) +
	       AllocatorBlockBytes(view.delta_fold_scales.Entries().capacity() * sizeof(superslm::SslmAmplifyingFoldEntry)) +
	       AllocatorBlockBytes(view.u_fold_scales.Entries().capacity() * sizeof(superslm::SslmAmplifyingFoldEntry)) +
	       AllocatorBlockBytes(layers.capacity() * sizeof(superslm::LayerAdapter)) +
	       StringHeapBytes(meta.source_adapter_name);
}

// ---- GPU packing (moved from sslm_gpu_adapter_mapImpl, unchanged) -------------------------------

// The prepared object (gpu_1p0.h's SslmGpuPreparedAdapter is this, with the model identity token).
struct PreparedAdapterCore {
	std::vector<uint8_t> ab_bytes;    // every covered slot's lora_A then lora_B rows
	std::vector<uint8_t> fold_bytes;  // every covered slot's DFS1 then UFS1 triples
	std::vector<std::array<superslm_gpu::AdapterProjSlot, 7>> slots;  // [layer][q,o,gate,up,down,k,v]
	uint32_t rank = 0;
	uint32_t covered_layers = 0;
	AdapterMeta meta;
	std::array<uint8_t, superslm::kIntegrityHashBytes> artifact_hash{};
};

inline const superslm::LayerAdapterProjection& ProjectionAt(const superslm::LayerAdapter& la, int p) noexcept {
	switch (p) {
		case 0: return la.q;
		case 1: return la.o;
		case 2: return la.gate;
		case 3: return la.up;
		case 4: return la.down;
		case 5: return la.k;
		default: return la.v;
	}
}

// Packs the populated `layers` into `out` exactly as the GPU map always has: q,o,gate,up,down,k,v
// within each layer, A then B into ab_bytes, DFS1 then UFS1 into fold_bytes. Both buffers are
// reserved to their final size before the first byte goes in (plan rev 1.2, re-check L5). An
// uncovered adapter still gets one byte in each, as the map's MakeBuffer/Upload need.
inline void PackAdapterSlots(const std::vector<superslm::LayerAdapter>& layers, const BaseModelGeometry& base,
                             PreparedAdapterCore& out) {
	static const char* const kProjNamesByIndex[7] = {"q_proj",   "o_proj", "gate_proj", "up_proj",
	                                                  "down_proj", "k_proj", "v_proj"};
	// PEAK-BOUND slots
	out.slots.assign(base.num_hidden_layers, std::array<superslm_gpu::AdapterProjSlot, 7>{});
	uint64_t ab_total = 0, fold_total = 0;
	for (uint32_t l = 0; l < base.num_hidden_layers; ++l) {
		const superslm::LayerAdapter& la = layers[l];
		for (int p = 0; p < 7; ++p) {
			const superslm::LayerAdapterProjection& proj = ProjectionAt(la, p);
			if (!proj.a_weight || !proj.b_weight || !proj.delta_fold_entry || !proj.u_fold_entry) continue;
			const uint64_t in_channels = AdapterInChannelsFor(kProjNamesByIndex[p], base.hidden_size,
			                                                  base.intermediate_size, base.q_width);
			ab_total += uint64_t(la.rank) * in_channels + proj.delta_fold_entry->row_count * la.rank;
			fold_total += (proj.delta_fold_entry->row_count + proj.u_fold_entry->row_count) * 12u;
		}
	}
	// PEAK-BOUND packed
	out.ab_bytes.reserve(ab_total ? ab_total : 1);
	out.fold_bytes.reserve(fold_total ? fold_total : 1);
	for (uint32_t l = 0; l < base.num_hidden_layers; ++l) {
		const superslm::LayerAdapter& la = layers[l];
		for (int p = 0; p < 7; ++p) {
			const superslm::LayerAdapterProjection& proj = ProjectionAt(la, p);
			if (!proj.a_weight || !proj.b_weight || !proj.delta_fold_entry || !proj.u_fold_entry) {
				continue;  // not covered: present stays false (forward_sites.h's B1c shape)
			}
			superslm_gpu::AdapterProjSlot& slot = out.slots[l][static_cast<size_t>(p)];
			slot.present = true;
			slot.rank = la.rank;
			slot.out_channels = static_cast<uint32_t>(proj.delta_fold_entry->row_count);
			// SSLM-GEOMETRY-SITE: GS-28
			// a_weight is [rank, in_channels]; AdapterInChannelsFor owns the projection-to-in_channels
			// rule (o_proj reads q_width, T-2509).
			const uint64_t in_channels = AdapterInChannelsFor(kProjNamesByIndex[p], base.hidden_size,
			                                                  base.intermediate_size, base.q_width);
			slot.a_bytes = static_cast<uint64_t>(la.rank) * in_channels;
			slot.b_bytes = static_cast<uint64_t>(slot.out_channels) * la.rank;
			slot.a_offset = out.ab_bytes.size();
			out.ab_bytes.insert(out.ab_bytes.end(), proj.a_weight, proj.a_weight + slot.a_bytes);
			slot.b_offset = out.ab_bytes.size();
			out.ab_bytes.insert(out.ab_bytes.end(), proj.b_weight, proj.b_weight + slot.b_bytes);
			slot.fold_offset = out.fold_bytes.size();
			const uint64_t dfs_bytes = static_cast<uint64_t>(proj.delta_fold_entry->row_count) * 12u;
			const uint64_t ufs_bytes = static_cast<uint64_t>(proj.u_fold_entry->row_count) * 12u;
			out.fold_bytes.insert(out.fold_bytes.end(), proj.delta_fold_entry->data,
			                      proj.delta_fold_entry->data + dfs_bytes);
			out.fold_bytes.insert(out.fold_bytes.end(), proj.u_fold_entry->data, proj.u_fold_entry->data + ufs_bytes);
		}
	}
	if (out.ab_bytes.empty()) out.ab_bytes.resize(1, 0);
	if (out.fold_bytes.empty()) out.fold_bytes.resize(1, 0);
}

// What a prepare reports. The callers map each to their own status (sslm_gpu_adapter_prepare:
// kRejected -> SSLM_GPU_ADAPTER_ARTIFACT_REJECTED, kBaseHashMismatch ->
// SSLM_ADAPTER_BASE_HASH_MISMATCH; an allocation failure is a thrown std::bad_alloc).
enum class PrepareResult { kOk, kRejected, kBaseHashMismatch };

// Validates and populates against an already-parsed view, then packs. `view` stays the caller's.
inline PrepareResult PrepareAdapterCoreFromView(const superslm::SslmModelView& view, const BaseModelGeometry& base,
                                                PreparedAdapterCore& out) {
	std::vector<superslm::LayerAdapter> layers;
	const AdapterLoadStatus st = PopulateAdapterFromView(view, base, out.meta, layers, nullptr);
	if (st == AdapterLoadStatus::BaseHashMismatch) return PrepareResult::kBaseHashMismatch;
	if (st != AdapterLoadStatus::Ok) return PrepareResult::kRejected;
	PackAdapterSlots(layers, base, out);
	out.rank = out.meta.rank;
	out.covered_layers = CoveredLayers(layers);
	out.artifact_hash = view.RawIntegrityHash();
	return PrepareResult::kOk;
}

// The device-free prepare from bytes: prescreen, SslmModel::Load (hash and copy), populate, pack.
// The parsed view (and so the artifact copy) is dropped on return; `out` keeps the packed bytes,
// the slot table and the info fields. Throws std::bad_alloc on an allocation failure, leaving
// `out` for the caller to discard.
inline PrepareResult PrepareAdapterCore(const BaseModelGeometry& base, const uint8_t* data, size_t size,
                                        PreparedAdapterCore& out) {
	if (!PrescreenAdapterArtifact(data, size, base.num_hidden_layers)) return PrepareResult::kRejected;
	superslm::SslmModelView view;
	if (superslm::SslmModel::Load(data, size, view, nullptr) != superslm::SslmModelStatus::Ok)
		return PrepareResult::kRejected;
	return PrepareAdapterCoreFromView(view, base, out);
}

// The prepared object's own heap bytes (the prepared info's host_resident_bytes, less the object).
inline uint64_t PreparedCoreHeapBytes(const PreparedAdapterCore& p) noexcept {
	return AllocatorBlockBytes(p.ab_bytes.capacity()) + AllocatorBlockBytes(p.fold_bytes.capacity()) +
	       AllocatorBlockBytes(p.slots.capacity() * sizeof(std::array<superslm_gpu::AdapterProjSlot, 7>)) +
	       StringHeapBytes(p.meta.source_adapter_name);
}

}  // namespace superslm_adapter

#endif  // SUPERSLM_SRC_ADAPTER_CORE_H
