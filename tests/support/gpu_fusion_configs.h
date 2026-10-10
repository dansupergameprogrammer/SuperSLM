// GPU per-token overhead plan rev 3.1, §3.2-3.3 (S2, S3): the bracket-mode fusion configurations
// every S2/S3 cell runs, with the per-layer dispatch counts §7 cells 7.3 and 7.8 assert for each.
//
// Bracket mode's recorder reads one `GpuBracketFusionConfig` (include/superslm/gpu_port.h): which
// of S2's two runs it fuses (o tail + attn_residual + mlp_norm; down tail + mlp_residual + commit)
// and which S3 variant it ships (none, the tail-only variant, or the merged gate+up GEMM). B0.7
// decides which configuration ships; the build compiles in the full plan as the default and the
// cells run every configuration that could ship, through `SetGpuBracketFusionConfig`. Per-site
// mode never fuses, whatever the configuration.
//
// The counts are written out here, never derived from the library: 7.8 ("the cell asserts the
// counts of the configuration the recorder was built for") and 7.3's killing value k = 12 are
// checked against this table, so a recorder that miscounts cannot also move its own expectation.
// Per layer, legacy / QK norm (plan §7 7.8, rev 3.1): unfused 24 / 25; one S2 run 22 / 23; both S2
// runs 20 / 21; S3 adds the merged GEMM's saving (-2) or the tail-only variant's (-1).
//
// Shared by tests/test_main.cpp (the synthetic cells) and tests/gpu-overhead-s1-suite/ (the
// real-artifact cells), so both run the same configurations.
#ifndef SSLM_TESTS_GPU_FUSION_CONFIGS_H
#define SSLM_TESTS_GPU_FUSION_CONFIGS_H

#include <cstdint>
#include <string>
#include <vector>

#include "superslm/gpu_port.h"

namespace superslm_gpu_fusion_configs {

using superslm_gpu::GpuBracketFusionConfig;
using superslm_gpu::GpuS3Fusion;

struct ConfigRow {
	const char* name;
	GpuBracketFusionConfig config;
	uint32_t legacy_per_layer;  // dispatches (and barriers) bracket mode records per legacy layer
	uint32_t qk_per_layer;      // ... per layer of a model that carries QK norm
	uint32_t PerLayer(bool has_qk_norm) const { return has_qk_norm ? qk_per_layer : legacy_per_layer; }
};

// The S1 recorder: no run fused. Still shippable (B0.7 may fuse neither run), and the
// configuration the S1 cells that assert S1's bracket counts and stream shape pin.
inline constexpr GpuBracketFusionConfig kUnfused{false, false, GpuS3Fusion::None};

// The compiled-in default at this slice: the full plan as far as it is built. At S2 that is both
// runs and no S3 variant.
inline constexpr GpuBracketFusionConfig kSliceDefault{true, true, GpuS3Fusion::None};

// Every configuration that can ship at this slice, with its counts.
inline const std::vector<ConfigRow>& ShippableConfigs() {
	static const std::vector<ConfigRow> rows = {
	    {"unfused", kUnfused, 24, 25},
	    {"S2 o run", GpuBracketFusionConfig{true, false, GpuS3Fusion::None}, 22, 23},
	    {"S2 down run", GpuBracketFusionConfig{false, true, GpuS3Fusion::None}, 22, 23},
	    {"S2 both runs", GpuBracketFusionConfig{true, true, GpuS3Fusion::None}, 20, 21},
	};
	return rows;
}

// Every configuration `SetGpuBracketFusionConfig` must refuse at this slice: at S2, every S3
// variant (no S3 recorder exists yet).
inline const std::vector<GpuBracketFusionConfig>& RefusedConfigs() {
	static const std::vector<GpuBracketFusionConfig> rows = [] {
		std::vector<GpuBracketFusionConfig> out;
		for (GpuS3Fusion s3 : {GpuS3Fusion::TailOnly, GpuS3Fusion::MergedGemm}) {
			for (int o = 0; o < 2; ++o) {
				for (int d = 0; d < 2; ++d) out.push_back(GpuBracketFusionConfig{o != 0, d != 0, s3});
			}
		}
		return out;
	}();
	return rows;
}

inline std::string ConfigName(const GpuBracketFusionConfig& c) {
	std::string s = "{o:";
	s += c.fuse_o_run ? "fused" : "-";
	s += " down:";
	s += c.fuse_down_run ? "fused" : "-";
	s += " s3:";
	s += c.s3 == GpuS3Fusion::MergedGemm ? "merged" : c.s3 == GpuS3Fusion::TailOnly ? "tail-only" : "-";
	s += "}";
	return s;
}

// Selects a configuration for one cell and restores the process's configuration on exit.
// `accepted` is SetGpuBracketFusionConfig's answer: a cell that asks for a refused configuration
// runs under the previous one, so it checks `accepted` first.
struct ScopedFusionConfig {
	GpuBracketFusionConfig previous;
	bool accepted;
	explicit ScopedFusionConfig(const GpuBracketFusionConfig& config)
	    : previous(superslm_gpu::GetGpuBracketFusionConfig()),
	      accepted(superslm_gpu::SetGpuBracketFusionConfig(config)) {}
	~ScopedFusionConfig() { superslm_gpu::SetGpuBracketFusionConfig(previous); }
	ScopedFusionConfig(const ScopedFusionConfig&) = delete;
	ScopedFusionConfig& operator=(const ScopedFusionConfig&) = delete;
};

}  // namespace superslm_gpu_fusion_configs

#endif  // SSLM_TESTS_GPU_FUSION_CONFIGS_H
