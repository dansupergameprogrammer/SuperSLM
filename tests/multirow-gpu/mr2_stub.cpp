// MR2 red suite: the stand-in for the recorder's entry and seams (mr2_entry.h) while src/ has none. Linked by
// mr2_wine.sh only when the engine defines no sslm_gpu_seq_embed_rows; the cells are then compiled with
// -DMR2_STUB and report red-unimplemented. Every function refuses with an argument-fault status and writes
// nothing, so no cell can pass against it.
#include "mr2_entry.h"

SslmGpuStatus sslm_gpu_seq_embed_rows(SslmGpuContext*, SslmGpuSequenceHandle*, const SslmGpuRowsSpec*,
                                      int32_t*) noexcept {
	return SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
}
SslmGpuStatus sslm_gpu_seq_rows_abandon(SslmGpuContext*, SslmGpuSequenceHandle*) noexcept {
	return SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
}
SslmGpuStatus sslm_gpu_seq_rows_open(const SslmGpuSequenceHandle*, int32_t*) noexcept {
	return SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
}
SslmGpuStatus sslm_gpu_seq_last_rows_committed(const SslmGpuSequenceHandle*, int32_t*) noexcept {
	return SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
}
SslmGpuStatus sslm_gpu_test_only_mr2_seams(SslmGpuContext*, SslmGpuMr2Seams*) noexcept {
	return SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
}
SslmGpuStatus sslm_gpu_test_only_mr2_chunk_state(const SslmGpuSequenceHandle*, SslmGpuMr2ChunkState*) noexcept {
	return SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
}
SslmGpuStatus sslm_gpu_test_only_mr2_device_kv(SslmGpuContext*, SslmGpuKvPoolHandle*, SslmGpuSequenceHandle*,
                                               std::vector<uint8_t>*) noexcept {
	return SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
}
SslmGpuStatus sslm_gpu_test_only_mr2_arm_slice_alloc_failure(SslmGpuContext*, int32_t) noexcept {
	return SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
}
