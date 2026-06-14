#ifndef QW3_SSD_H
#define QW3_SSD_H

#include <stdbool.h>
#include <stdint.h>

#define QW3_QK_K 256

typedef struct {
    void *ptr;
    uint64_t bytes;
} qw3_ssd_memory_lock;

typedef struct {
    uint64_t model_target_bytes;
    uint64_t cache_bytes;
    uint64_t effective_cache_bytes;
    uint32_t cache_experts;
} qw3_ssd_cache_plan;

bool qw3_parse_gib_arg(const char *s, uint64_t *bytes);
bool qw3_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes);

uint32_t qw3_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes);
bool qw3_ssd_auto_cache_plan(uint64_t            recommended_bytes,
                             uint64_t            non_routed_bytes,
                             uint64_t            per_expert_bytes,
                             uint64_t            max_model_experts,
                             qw3_ssd_cache_plan *out);

bool qw3_ssd_memory_lock_acquire(qw3_ssd_memory_lock *lock,
                                 uint64_t             bytes);
void qw3_ssd_memory_lock_release(qw3_ssd_memory_lock *lock);

#endif
