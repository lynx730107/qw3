#include "optimization_acceptance.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    qw3_opt_error rounded={0},bad={0},nonfinite={0},signed_zero={0};
    float a=100.0f,b=nextafterf(nextafterf(a,INFINITY),INFINITY);
    qw3_opt_error_add(&rounded,a,b);
    assert(rounded.different_bits==1 && qw3_opt_kernel_valid(&rounded) && qw3_opt_logits_valid(&rounded));
    qw3_opt_error_add(&bad,a,101.0f);
    assert(!qw3_opt_kernel_valid(&bad) && !qw3_opt_logits_valid(&bad));
    qw3_opt_error_add(&nonfinite,1,NAN);
    assert(!qw3_opt_kernel_valid(&nonfinite) && !qw3_opt_logits_valid(&nonfinite));
    qw3_opt_error_add(&signed_zero,0.0f,-0.0f);
    assert(signed_zero.different_bits==1 && qw3_opt_kernel_valid(&signed_zero) && qw3_opt_logits_valid(&signed_zero));
    puts("PASS optimization_acceptance_v2 rounding=accepted substantive_error=rejected NaN=rejected signed_zero=accepted");
    return 0;
}
