#ifndef QW3_VISION_H
#define QW3_VISION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qw3_vision qw3_vision;

typedef struct {
    float *embd;        /* [n_tokens * embd_dim] float32, caller frees via qw3_vision_result_free */
    int n_tokens;       /* number of visual tokens produced by projector */
    int embd_dim;       /* output embedding dim (2304 for Inkling-2-Mini) */
    int grid_w;         /* visual token grid width */
    int grid_h;         /* visual token grid height */
    int proc_w;         /* preprocessed padded image width (multiple of patch_size * merge) */
    int proc_h;         /* preprocessed padded image height (multiple of patch_size * merge) */
    double encode_ms;   /* total preprocess + GPU encode time in ms */
} qw3_vision_result;

/* Open a vision mmproj GGUF (currently supports inkling2v).
 * max_tokens <= 0 uses the model default (4096 for Inkling-2-Mini).
 * Returns 0 on success, -1 on error (with message in err). */
int qw3_vision_open(qw3_vision **out, const char *mmproj_path,
                    int max_tokens, char *err, size_t errlen);

void qw3_vision_close(qw3_vision *v);

const char *qw3_vision_arch(const qw3_vision *v);
int qw3_vision_embd_dim(const qw3_vision *v);
int qw3_vision_max_tokens(const qw3_vision *v);

/* Encode an image from raw encoded bytes (PNG/JPEG/WebP/GIF/HEIC/BMP/TIFF). */
int qw3_vision_encode_bytes(qw3_vision *v, const uint8_t *bytes, size_t len,
                            qw3_vision_result *out, char *err, size_t errlen);

/* Encode an image from a local file path. */
int qw3_vision_encode_file(qw3_vision *v, const char *path,
                           qw3_vision_result *out, char *err, size_t errlen);

/* Encode an already-decoded 8-bit RGB image (width * height * 3 bytes, row-major). */
int qw3_vision_encode_rgb(qw3_vision *v, const uint8_t *rgb,
                          int width, int height,
                          qw3_vision_result *out, char *err, size_t errlen);

void qw3_vision_result_free(qw3_vision_result *r);

#ifdef __cplusplus
}
#endif

#endif /* QW3_VISION_H */
