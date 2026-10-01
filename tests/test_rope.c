/* No model required: Qwen text IMRoPE pairs the two halves of the rotated
 * prefix. Position zero alone cannot detect an adjacent-pair regression. */
#include "../qw3.c"

int main(void) {
    const int positions[] = {0, 1, 28, 511};
    const int dimensions[] = {128, 256};
    const float thetas[] = {10000.0f, 10000000.0f};
    const int heads = 3, rotated = 64;
    float input[768], cpu[768], reference[768];
#ifndef QW3_NO_METAL
    float gpu[768];
#endif
    int checks = 0;
    for (int d = 0; d < 2; d++) {
        const int hd = dimensions[d], n = heads * hd;
        for (int i = 0; i < n; i++) input[i] = ((i * 17) % 101 - 50) / 31.0f;
        for (int t = 0; t < 2; t++) for (int p = 0; p < 4; p++) {
            memcpy(cpu, input, (size_t)n * sizeof(float));
            memcpy(reference, input, (size_t)n * sizeof(float));
            for (int h = 0; h < heads; h++) {
                cpu_rope_head(cpu + h * hd, positions[p], rotated, thetas[t]);
                for (int i = 0; i < rotated / 2; i++) {
                    const double angle = positions[p] * pow((double)thetas[t], -2.0 * i / rotated);
                    const double x0 = input[h * hd + i], x1 = input[h * hd + i + rotated / 2];
                    reference[h * hd + i] = (float)(x0 * cos(angle) - x1 * sin(angle));
                    reference[h * hd + i + rotated / 2] = (float)(x0 * sin(angle) + x1 * cos(angle));
                }
            }
#ifndef QW3_NO_METAL
            if (!qw3_metal_rope_heads(input, heads, hd, rotated, positions[p], thetas[t], gpu)) return 1;
#endif
            double cpu_max = 0.0, gpu_max = 0.0;
            for (int i = 0; i < n; i++) {
                const double diff = fabs((double)cpu[i] - reference[i]);
                if (!isfinite(cpu[i]) || diff > 0.0001 ||
                    (i % hd >= rotated && cpu[i] != input[i])) return 1;
                if (diff > cpu_max) cpu_max = diff;
#ifndef QW3_NO_METAL
                const double gd = fabs((double)gpu[i] - reference[i]);
                if (!isfinite(gpu[i]) || gd > 0.0001 ||
                    (i % hd >= rotated && gpu[i] != input[i])) return 1;
                if (gd > gpu_max) gpu_max = gd;
#endif
            }
            printf("rope split-half: ok head_dim=%d theta=%.0f pos=%d cpu_max=%.9g gpu_max=%.9g\n",
                   hd, thetas[t], positions[p], cpu_max, gpu_max);
            checks++;
        }
    }
    printf("test-rope: ok checks=%d\n", checks);
    return 0;
}
