#ifndef SUPERSLM_INCLUDE_ADAPTER_INFO_H
#define SUPERSLM_INCLUDE_ADAPTER_INFO_H
/* What a mapped (or, on the GPU, prepared) LoRA adapter is and what it costs: its ADP1 fields, its
 * hashes and the exact bytes it holds. Filled by sslm_adapter_info_get (CPU, sslm_abi.h) and by
 * sslm_gpu_adapter_info_get / sslm_gpu_prepared_adapter_info_get (gpu_1p0.h). C, no dependencies,
 * so both headers include it.
 *
 * The caller sets struct_size to sizeof(sslm_adapter_info) (any size at or above the v1 size,
 * SSLM_ADAPTER_INFO_V1_SIZE, is accepted; bytes past the v1 size are left untouched). A smaller size
 * or a null argument is refused (CPU SSLM_INVALID_ARGUMENT, GPU SSLM_GPU_INVALID_ARGUMENT). */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sslm_adapter_info_s {
    uint32_t struct_size;            /* caller sets; forward-compatible */
    uint32_t adp1_version;
    uint32_t rank;
    uint32_t target_modules_mask;    /* ADP1 bits 0..6: q,o,gate,up,down,k,v */
    uint32_t covered_layers;         /* base layers with at least one adapted projection */
    uint32_t use_rslora;             /* 0 or 1 */
    double   lora_alpha;
    uint8_t  base_artifact_hash[32];
    uint8_t  artifact_hash[32];      /* the adapter's own integrity hash (the .sslm header's) */
    uint64_t host_resident_bytes;    /* every host heap byte the handle (or prepared object) owns */
    uint64_t device_resident_bytes;  /* GPU handle: the buffers' allocation sizes. Prepared object:
                                      * the predicted allocation (each buffer rounded up to 64 KiB).
                                      * CPU: 0 */
    uint64_t upload_staging_bytes;   /* prepared object only: UPLOAD-heap bytes map_prepared will
                                      * allocate (each buffer rounded up to 64 KiB); 0 otherwise */
    char     name[64];               /* ADP1 name, truncated to 63 bytes, NUL-terminated */
} sslm_adapter_info;

#define SSLM_ADAPTER_INFO_V1_SIZE ((uint32_t)sizeof(sslm_adapter_info))

#ifdef __cplusplus
}
#endif

#endif /* SUPERSLM_INCLUDE_ADAPTER_INFO_H */
