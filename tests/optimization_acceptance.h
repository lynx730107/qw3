#ifndef QW3_OPTIMIZATION_ACCEPTANCE_H
#define QW3_OPTIMIZATION_ACCEPTANCE_H

/* Policy v2: numerical correctness followed by functional agent acceptance.
 * These ceilings are initial screening limits, never sufficient for adoption.
 * Logit limits retain the existing dense-vs-llama differential-test limits.
 * Bit differences are diagnostic; immutable inputs/guards/history remain exact.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#define QW3_OPT_ACCEPTANCE_VERSION 2
#define QW3_OPT_KERNEL_NMSE_MAX 1e-7
#define QW3_OPT_STATE_NMSE_MAX 1e-7
#define QW3_OPT_LOGITS_RMSE_MAX 1e-3
#define QW3_OPT_LOGITS_MAX_ABS 5e-3

typedef struct {
    uint64_t count, different_bits;
    double squared_error, reference_energy, max_abs;
    int nonfinite;
} qw3_opt_error;

static inline void qw3_opt_error_add(qw3_opt_error *s,float reference,float actual) {
    uint32_t a,b;memcpy(&a,&reference,4);memcpy(&b,&actual,4);
    s->count++;s->different_bits+=(a!=b);
    if (!isfinite(reference) || !isfinite(actual)) {s->nonfinite=1;return;}
    double error=(double)actual-reference;
    s->squared_error+=error*error;
    s->reference_energy+=(double)reference*reference;
    if (fabs(error)>s->max_abs) s->max_abs=fabs(error);
}
static inline double qw3_opt_error_nmse(const qw3_opt_error *s) {
    if (s->nonfinite || !s->count) return INFINITY;
    if (!s->reference_energy) return s->squared_error ? INFINITY : 0;
    return s->squared_error/s->reference_energy;
}
static inline double qw3_opt_error_rmse(const qw3_opt_error *s) {
    return s->nonfinite || !s->count ? INFINITY : sqrt(s->squared_error/s->count);
}
static inline int qw3_opt_kernel_valid(const qw3_opt_error *s) {
    return qw3_opt_error_nmse(s)<=QW3_OPT_KERNEL_NMSE_MAX;
}
static inline int qw3_opt_logits_valid(const qw3_opt_error *s) {
    return !s->nonfinite && s->count &&
        qw3_opt_error_rmse(s)<=QW3_OPT_LOGITS_RMSE_MAX &&
        s->max_abs<=QW3_OPT_LOGITS_MAX_ABS;
}
#endif
