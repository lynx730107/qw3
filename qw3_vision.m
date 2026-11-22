/* Native Metal Vision Encoder for Inkling-2-Mini (inkling2v mmproj GGUF).
 * Zero external dependencies: uses Apple Metal, ImageIO, and CoreGraphics. */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#import <mach-o/dyld.h>

#include "qw3_vision.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

/* ============================================================================
 * Minimal GGUF v2/v3 parser for mmproj files
 * ============================================================================ */

enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

enum {
    GGML_T_F32  = 0,
    GGML_T_F16  = 1,
    GGML_T_BF16 = 30,
};

typedef struct {
    const uint8_t *base;
    size_t size;
    size_t pos;
} gguf_reader;

static bool rd_bytes(gguf_reader *r, void *dst, size_t n) {
    if (r->pos > r->size || n > r->size - r->pos) return false;
    if (dst) memcpy(dst, r->base + r->pos, n);
    r->pos += n;
    return true;
}

static bool rd_u32(gguf_reader *r, uint32_t *out) { return rd_bytes(r, out, 4); }
static bool rd_i32(gguf_reader *r, int32_t *out)  { return rd_bytes(r, out, 4); }
static bool rd_u64(gguf_reader *r, uint64_t *out) { return rd_bytes(r, out, 8); }
static bool rd_f32(gguf_reader *r, float *out)    { return rd_bytes(r, out, 4); }

static bool rd_str(gguf_reader *r, char **out) {
    uint64_t n = 0;
    if (!rd_u64(r, &n) || n > (1u << 24) || n > r->size - r->pos) return false;
    if (out) {
        char *s = (char *)malloc((size_t)n + 1);
        if (!s) return false;
        memcpy(s, r->base + r->pos, (size_t)n);
        s[n] = '\0';
        *out = s;
    }
    r->pos += (size_t)n;
    return true;
}

static bool rd_skip_val(gguf_reader *r, uint32_t t) {
    switch (t) {
    case GGUF_TYPE_UINT8:
    case GGUF_TYPE_INT8:
    case GGUF_TYPE_BOOL:    return rd_bytes(r, NULL, 1);
    case GGUF_TYPE_UINT16:
    case GGUF_TYPE_INT16:   return rd_bytes(r, NULL, 2);
    case GGUF_TYPE_UINT32:
    case GGUF_TYPE_INT32:
    case GGUF_TYPE_FLOAT32: return rd_bytes(r, NULL, 4);
    case GGUF_TYPE_UINT64:
    case GGUF_TYPE_INT64:
    case GGUF_TYPE_FLOAT64: return rd_bytes(r, NULL, 8);
    case GGUF_TYPE_STRING:  return rd_str(r, NULL);
    case GGUF_TYPE_ARRAY: {
        uint32_t et = 0;
        uint64_t n = 0;
        if (!rd_u32(r, &et) || !rd_u64(r, &n)) return false;
        for (uint64_t i = 0; i < n; i++) {
            if (!rd_skip_val(r, et)) return false;
        }
        return true;
    }
    default: return false;
    }
}

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset; /* relative to tensor data base; adjusted to file offset after header */
} vis_tensor;

typedef struct {
    vis_tensor ln1_w, ln1_b;
    vis_tensor ln2_w, ln2_b;
    vis_tensor ls1_w, ls2_w;
    vis_tensor q_w, q_b;
    vis_tensor k_w, k_b;
    vis_tensor v_w, v_b;
    vis_tensor o_w, o_b;
    vis_tensor ff_up_w, ff_up_b;
    vis_tensor ff_down_w, ff_down_b;
} vis_layer;

/* Uniform structs matching metal/qw3_vision.metal */
typedef struct {
    uint32_t W, H, P, W_p, H_p, D, pos_grid, w_type, has_bias;
} vis_patch_args;

typedef struct {
    uint32_t N, D;
    float eps;
} vis_ln_args;

typedef struct {
    uint32_t M, N_out, K, w_type, has_bias, act_gelu;
} vis_gemm_args;

typedef struct {
    uint32_t N, W_p, n_head, d_head;
    float theta;
} vis_rope_args;

typedef struct {
    uint32_t N, n_head, d_head;
    float scale;
} vis_attn_args;

typedef struct {
    uint32_t N, D;
} vis_res_args;

typedef struct {
    uint32_t W_p, H_p, D, merge;
} vis_merge_args;

@interface QW3VisionObj : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) id<MTLBuffer> weightsBuf;
@property (nonatomic, strong) id<MTLComputePipelineState> pPatchEmbed;
@property (nonatomic, strong) id<MTLComputePipelineState> pLayerNorm;
@property (nonatomic, strong) id<MTLComputePipelineState> pGemmF16;
@property (nonatomic, strong) id<MTLComputePipelineState> pGemmGeneric;
@property (nonatomic, strong) id<MTLComputePipelineState> pRope2D;
@property (nonatomic, strong) id<MTLComputePipelineState> pAttnFlash64;
@property (nonatomic, strong) id<MTLComputePipelineState> pResScale;
@property (nonatomic, strong) id<MTLComputePipelineState> pPatchMerge;

/* Reusable scratch buffers grown on demand */
@property (nonatomic, strong) id<MTLBuffer> bufImg;
@property (nonatomic, strong) id<MTLBuffer> bufX;
@property (nonatomic, strong) id<MTLBuffer> bufH;
@property (nonatomic, strong) id<MTLBuffer> bufQ;
@property (nonatomic, strong) id<MTLBuffer> bufK;
@property (nonatomic, strong) id<MTLBuffer> bufV;
@property (nonatomic, strong) id<MTLBuffer> bufAttn;
@property (nonatomic, strong) id<MTLBuffer> bufFF;
@property (nonatomic, strong) id<MTLBuffer> bufMerge;
@property (nonatomic, strong) id<MTLBuffer> bufP1;
@property (nonatomic, strong) id<MTLBuffer> bufOut;
@end

@implementation QW3VisionObj
@end

struct qw3_vision {
    void *obj; /* retained QW3VisionObj* */

    /* mmap backing */
    void *mmap_ptr;
    size_t mmap_len;
    int fd;

    /* Hyperparameters */
    char projector_type[64];
    int patch_size;
    int n_merge;
    int n_embd;
    int n_ff;
    int n_layer;
    int n_head;
    int d_head;
    int proj_hidden;
    int proj_dim;
    int pos_grid;
    int max_tokens;
    float eps;
    float rope_theta;
    float image_mean[3];
    float image_std[3];

    /* Tensors */
    vis_tensor class_embd;
    vis_tensor class_pos_embd;
    vis_tensor patch_w;
    vis_tensor patch_b;
    bool has_patch_b;
    vis_tensor pos_embd;
    vis_tensor pre_ln_w, pre_ln_b;
    vis_tensor mm1_w, mm1_b;
    bool has_mm1_b;
    vis_tensor mm2_w, mm2_b;
    bool has_mm2_b;
    vis_layer *layers;
};

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static NSString *load_vision_shader_source(char *err, size_t errlen) {
    NSMutableArray<NSString *> *candidates = [NSMutableArray array];
    const char *env_dir = getenv("QW3_METAL_KERNEL_DIR");
    if (env_dir && env_dir[0]) {
        [candidates addObject:[[NSString stringWithUTF8String:env_dir]
                               stringByAppendingPathComponent:@"qw3_vision.metal"]];
    }
    [candidates addObject:@"metal/qw3_vision.metal"];
    [candidates addObject:@"../metal/qw3_vision.metal"];

    uint32_t exe_sz = 0;
    _NSGetExecutablePath(NULL, &exe_sz);
    if (exe_sz > 0 && exe_sz < 4096) {
        char *exe = (char *)malloc(exe_sz + 1);
        if (exe && _NSGetExecutablePath(exe, &exe_sz) == 0) {
            NSString *exe_path = [[NSString stringWithUTF8String:exe] stringByResolvingSymlinksInPath];
            NSString *exe_dir = [exe_path stringByDeletingLastPathComponent];
            [candidates addObject:[exe_dir stringByAppendingPathComponent:@"metal/qw3_vision.metal"]];
            [candidates addObject:[[exe_dir stringByDeletingLastPathComponent]
                                   stringByAppendingPathComponent:@"metal/qw3_vision.metal"]];
        }
        free(exe);
    }

    for (NSString *path in candidates) {
        NSError *readErr = nil;
        NSString *src = [NSString stringWithContentsOfFile:path
                                                  encoding:NSUTF8StringEncoding
                                                     error:&readErr];
        if (src) return src;
    }
    if (err && errlen) {
        snprintf(err, errlen, "could not locate metal/qw3_vision.metal");
    }
    return nil;
}

static const vis_tensor *find_tensor(const vis_tensor *arr, size_t n, const char *name) {
    for (size_t i = 0; i < n; i++) {
        if (strcmp(arr[i].name, name) == 0) return &arr[i];
    }
    return NULL;
}

static bool require_tensor(const vis_tensor *arr, size_t n, const char *name,
                           vis_tensor *out, char *err, size_t errlen) {
    const vis_tensor *t = find_tensor(arr, n, name);
    if (!t) {
        if (err && errlen) snprintf(err, errlen, "mmproj missing required tensor '%s'", name);
        return false;
    }
    if (t->type != GGML_T_F32 && t->type != GGML_T_F16 && t->type != GGML_T_BF16) {
        if (err && errlen) snprintf(err, errlen, "mmproj tensor '%s' has unsupported type %u", name, t->type);
        return false;
    }
    *out = *t;
    out->name = NULL;
    return true;
}

static bool optional_tensor(const vis_tensor *arr, size_t n, const char *name,
                            vis_tensor *out, bool *present) {
    const vis_tensor *t = find_tensor(arr, n, name);
    if (!t) {
        *present = false;
        return true;
    }
    *out = *t;
    out->name = NULL;
    *present = true;
    return true;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> dev, id<MTLLibrary> lib,
                                                 NSString *name, char *err, size_t errlen) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        if (err && errlen) snprintf(err, errlen, "missing vision Metal kernel '%s'", name.UTF8String);
        return nil;
    }
    NSError *pe = nil;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&pe];
    if (!pso && err && errlen) {
        snprintf(err, errlen, "failed to compile vision pipeline '%s': %s",
                 name.UTF8String, pe ? pe.localizedDescription.UTF8String : "unknown");
    }
    return pso;
}

int qw3_vision_open(qw3_vision **out, const char *mmproj_path,
                    int max_tokens, char *err, size_t errlen) {
    if (!out || !mmproj_path) {
        if (err && errlen) snprintf(err, errlen, "invalid vision open arguments");
        return -1;
    }
    *out = NULL;

    @autoreleasepool {
        int fd = open(mmproj_path, O_RDONLY);
        if (fd < 0) {
            if (err && errlen) snprintf(err, errlen, "cannot open mmproj '%s': %s", mmproj_path, strerror(errno));
            return -1;
        }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 64) {
            close(fd);
            if (err && errlen) snprintf(err, errlen, "invalid mmproj file '%s'", mmproj_path);
            return -1;
        }
        size_t file_sz = (size_t)st.st_size;
        void *mapped = mmap(NULL, file_sz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            close(fd);
            if (err && errlen) snprintf(err, errlen, "mmap failed for '%s': %s", mmproj_path, strerror(errno));
            return -1;
        }

        qw3_vision *v = (qw3_vision *)calloc(1, sizeof(*v));
        if (!v) {
            munmap(mapped, file_sz);
            close(fd);
            return -1;
        }
        v->mmap_ptr = mapped;
        v->mmap_len = file_sz;
        v->fd = fd;

        /* Defaults */
        v->patch_size = 20;
        v->n_merge = 2;
        v->n_embd = 1024;
        v->n_ff = 4096;
        v->n_layer = 23;
        v->n_head = 16;
        v->proj_dim = 2304;
        v->eps = 1e-5f;
        v->rope_theta = 10000.0f;
        v->image_mean[0] = v->image_mean[1] = v->image_mean[2] = 0.5f;
        v->image_std[0]  = v->image_std[1]  = v->image_std[2]  = 0.5f;
        v->max_tokens = (max_tokens > 0) ? max_tokens : 4096;

        gguf_reader r = { (const uint8_t *)mapped, file_sz, 0 };
        char magic[4];
        uint32_t version = 0;
        uint64_t n_tensors = 0, n_kv = 0;
        if (!rd_bytes(&r, magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
            !rd_u32(&r, &version) || (version != 2 && version != 3) ||
            !rd_u64(&r, &n_tensors) || !rd_u64(&r, &n_kv)) {
            if (err && errlen) snprintf(err, errlen, "'%s' is not a valid GGUF v2/v3 file", mmproj_path);
            qw3_vision_close(v);
            return -1;
        }

        uint32_t alignment = 32;
        for (uint64_t i = 0; i < n_kv; i++) {
            char *key = NULL;
            uint32_t vt = 0;
            if (!rd_str(&r, &key) || !rd_u32(&r, &vt)) {
                free(key);
                if (err && errlen) snprintf(err, errlen, "corrupt GGUF KV metadata");
                qw3_vision_close(v);
                return -1;
            }
            bool handled = true;
            if (strcmp(key, "general.alignment") == 0 && vt == GGUF_TYPE_UINT32) {
                rd_u32(&r, &alignment);
            } else if (strcmp(key, "clip.vision.projector_type") == 0 && vt == GGUF_TYPE_STRING) {
                char *s = NULL;
                if (rd_str(&r, &s)) {
                    snprintf(v->projector_type, sizeof(v->projector_type), "%s", s);
                    free(s);
                }
            } else if (strcmp(key, "clip.vision.patch_size") == 0 && (vt == GGUF_TYPE_UINT32 || vt == GGUF_TYPE_INT32)) {
                int32_t val = 0; rd_i32(&r, &val); v->patch_size = val;
            } else if (strcmp(key, "clip.vision.spatial_merge_size") == 0 && (vt == GGUF_TYPE_UINT32 || vt == GGUF_TYPE_INT32)) {
                int32_t val = 0; rd_i32(&r, &val); v->n_merge = val;
            } else if (strcmp(key, "clip.vision.embedding_length") == 0 && (vt == GGUF_TYPE_UINT32 || vt == GGUF_TYPE_INT32)) {
                int32_t val = 0; rd_i32(&r, &val); v->n_embd = val;
            } else if (strcmp(key, "clip.vision.feed_forward_length") == 0 && (vt == GGUF_TYPE_UINT32 || vt == GGUF_TYPE_INT32)) {
                int32_t val = 0; rd_i32(&r, &val); v->n_ff = val;
            } else if (strcmp(key, "clip.vision.block_count") == 0 && (vt == GGUF_TYPE_UINT32 || vt == GGUF_TYPE_INT32)) {
                int32_t val = 0; rd_i32(&r, &val); v->n_layer = val;
            } else if (strcmp(key, "clip.vision.attention.head_count") == 0 && (vt == GGUF_TYPE_UINT32 || vt == GGUF_TYPE_INT32)) {
                int32_t val = 0; rd_i32(&r, &val); v->n_head = val;
            } else if (strcmp(key, "clip.vision.projection_dim") == 0 && (vt == GGUF_TYPE_UINT32 || vt == GGUF_TYPE_INT32)) {
                int32_t val = 0; rd_i32(&r, &val); v->proj_dim = val;
            } else if (strcmp(key, "clip.vision.attention.layer_norm_epsilon") == 0 && vt == GGUF_TYPE_FLOAT32) {
                rd_f32(&r, &v->eps);
            } else if ((strcmp(key, "clip.vision.image_mean") == 0 || strcmp(key, "clip.vision.image_std") == 0) &&
                       vt == GGUF_TYPE_ARRAY) {
                uint32_t et = 0;
                uint64_t n = 0;
                if (rd_u32(&r, &et) && rd_u64(&r, &n) && et == GGUF_TYPE_FLOAT32 && n == 3) {
                    float *dst = (strcmp(key, "clip.vision.image_mean") == 0) ? v->image_mean : v->image_std;
                    for (int c = 0; c < 3; c++) rd_f32(&r, &dst[c]);
                } else {
                    for (uint64_t j = 0; j < n; j++) rd_skip_val(&r, et);
                }
            } else {
                handled = false;
            }
            if (!handled) {
                if (!rd_skip_val(&r, vt)) {
                    free(key);
                    if (err && errlen) snprintf(err, errlen, "failed skipping GGUF metadata value");
                    qw3_vision_close(v);
                    return -1;
                }
            }
            free(key);
        }

        if (strcmp(v->projector_type, "inkling2v") != 0) {
            if (err && errlen) {
                snprintf(err, errlen, "unsupported mmproj projector_type '%s' (expected 'inkling2v')",
                         v->projector_type[0] ? v->projector_type : "(none)");
            }
            qw3_vision_close(v);
            return -1;
        }
        if (v->n_head <= 0 || (v->n_embd % v->n_head) != 0) {
            if (err && errlen) snprintf(err, errlen, "invalid vision head configuration (%d / %d)", v->n_embd, v->n_head);
            qw3_vision_close(v);
            return -1;
        }
        v->d_head = v->n_embd / v->n_head;
        if (v->d_head != 64) {
            if (err && errlen) snprintf(err, errlen, "vision d_head=%d not supported (expected 64)", v->d_head);
            qw3_vision_close(v);
            return -1;
        }

        /* Parse tensor table */
        vis_tensor *tensors = (vis_tensor *)calloc((size_t)n_tensors, sizeof(vis_tensor));
        if (!tensors) {
            qw3_vision_close(v);
            return -1;
        }
        bool tok = true;
        for (uint64_t i = 0; i < n_tensors; i++) {
            vis_tensor *t = &tensors[i];
            if (!rd_str(&r, &t->name) || !rd_u32(&r, &t->n_dims) || t->n_dims > 4) {
                tok = false;
                break;
            }
            for (uint32_t d = 0; d < t->n_dims; d++) {
                if (!rd_u64(&r, &t->dims[d])) { tok = false; break; }
            }
            if (!tok || !rd_u32(&r, &t->type) || !rd_u64(&r, &t->offset)) {
                tok = false;
                break;
            }
        }
        if (!tok) {
            for (uint64_t i = 0; i < n_tensors; i++) free(tensors[i].name);
            free(tensors);
            if (err && errlen) snprintf(err, errlen, "corrupt GGUF tensor directory");
            qw3_vision_close(v);
            return -1;
        }

        if (alignment == 0) alignment = 32;
        uint64_t data_base = ((uint64_t)r.pos + alignment - 1ull) & ~((uint64_t)alignment - 1ull);
        for (uint64_t i = 0; i < n_tensors; i++) {
            tensors[i].offset += data_base;
        }

        /* Bind all required inkling2v tensors */
        bool ok =
            require_tensor(tensors, (size_t)n_tensors, "v.class_embd", &v->class_embd, err, errlen) &&
            require_tensor(tensors, (size_t)n_tensors, "v.class_position_embd.weight", &v->class_pos_embd, err, errlen) &&
            require_tensor(tensors, (size_t)n_tensors, "v.patch_embd.weight", &v->patch_w, err, errlen) &&
            optional_tensor(tensors, (size_t)n_tensors, "v.patch_embd.bias", &v->patch_b, &v->has_patch_b) &&
            require_tensor(tensors, (size_t)n_tensors, "v.position_embd.weight", &v->pos_embd, err, errlen) &&
            require_tensor(tensors, (size_t)n_tensors, "v.pre_ln.weight", &v->pre_ln_w, err, errlen) &&
            require_tensor(tensors, (size_t)n_tensors, "v.pre_ln.bias", &v->pre_ln_b, err, errlen) &&
            require_tensor(tensors, (size_t)n_tensors, "mm.1.weight", &v->mm1_w, err, errlen) &&
            optional_tensor(tensors, (size_t)n_tensors, "mm.1.bias", &v->mm1_b, &v->has_mm1_b) &&
            require_tensor(tensors, (size_t)n_tensors, "mm.2.weight", &v->mm2_w, err, errlen) &&
            optional_tensor(tensors, (size_t)n_tensors, "mm.2.bias", &v->mm2_b, &v->has_mm2_b);

        if (ok) {
            v->pos_grid = (int)llround(sqrt((double)v->pos_embd.dims[1]));
            v->proj_hidden = (int)v->mm1_w.dims[1];
            v->proj_dim = (int)v->mm2_w.dims[1];
            v->layers = (vis_layer *)calloc((size_t)v->n_layer, sizeof(vis_layer));
            if (!v->layers) ok = false;
            for (int il = 0; ok && il < v->n_layer; il++) {
                char nm[96];
                vis_layer *L = &v->layers[il];
                #define REQ_L(field, suffix) do { \
                    snprintf(nm, sizeof(nm), "v.blk.%d." suffix, il); \
                    if (!require_tensor(tensors, (size_t)n_tensors, nm, &L->field, err, errlen)) ok = false; \
                } while (0)
                REQ_L(ln1_w,     "ln1.weight");
                REQ_L(ln1_b,     "ln1.bias");
                REQ_L(ln2_w,     "ln2.weight");
                REQ_L(ln2_b,     "ln2.bias");
                REQ_L(ls1_w,     "ls1.weight");
                REQ_L(ls2_w,     "ls2.weight");
                REQ_L(q_w,       "attn_q.weight");
                REQ_L(q_b,       "attn_q.bias");
                REQ_L(k_w,       "attn_k.weight");
                REQ_L(k_b,       "attn_k.bias");
                REQ_L(v_w,       "attn_v.weight");
                REQ_L(v_b,       "attn_v.bias");
                REQ_L(o_w,       "attn_out.weight");
                REQ_L(o_b,       "attn_out.bias");
                REQ_L(ff_up_w,   "ffn_up.weight");
                REQ_L(ff_up_b,   "ffn_up.bias");
                REQ_L(ff_down_w, "ffn_down.weight");
                REQ_L(ff_down_b, "ffn_down.bias");
                #undef REQ_L
            }
        }

        for (uint64_t i = 0; i < n_tensors; i++) free(tensors[i].name);
        free(tensors);
        if (!ok) {
            qw3_vision_close(v);
            return -1;
        }

        /* Initialize Metal device, shader library, pipelines, and weight buffer */
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            if (err && errlen) snprintf(err, errlen, "Metal device unavailable for vision encoder");
            qw3_vision_close(v);
            return -1;
        }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSString *src = load_vision_shader_source(err, errlen);
        if (!queue || !src) {
            qw3_vision_close(v);
            return -1;
        }
        NSError *ce = nil;
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeFast;
        }
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&ce];
        if (!lib) {
            if (err && errlen) {
                snprintf(err, errlen, "failed to compile qw3_vision.metal: %s",
                         ce ? ce.localizedDescription.UTF8String : "unknown");
            }
            qw3_vision_close(v);
            return -1;
        }

        QW3VisionObj *obj = [[QW3VisionObj alloc] init];
        obj.device = dev;
        obj.queue = queue;

        /* Zero-copy page-aligned Metal view of mmap'd mmproj weights */
        size_t page_sz = (size_t)getpagesize();
        size_t aligned_len = (file_sz + page_sz - 1) & ~(page_sz - 1);
        obj.weightsBuf = [dev newBufferWithBytesNoCopy:mapped
                                                length:aligned_len
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
        if (!obj.weightsBuf) {
            obj.weightsBuf = [dev newBufferWithBytes:mapped
                                              length:file_sz
                                             options:MTLResourceStorageModeShared];
        }
        if (!obj.weightsBuf) {
            if (err && errlen) snprintf(err, errlen, "failed to create Metal weight buffer for mmproj");
            qw3_vision_close(v);
            return -1;
        }

        obj.pPatchEmbed  = make_pipeline(dev, lib, @"qw3_vis_patch_embed", err, errlen);
        obj.pLayerNorm   = make_pipeline(dev, lib, @"qw3_vis_layernorm", err, errlen);
        obj.pGemmF16     = make_pipeline(dev, lib, @"qw3_vis_gemm_f16", err, errlen);
        obj.pGemmGeneric = make_pipeline(dev, lib, @"qw3_vis_gemm_generic", err, errlen);
        obj.pRope2D      = make_pipeline(dev, lib, @"qw3_vis_rope2d", err, errlen);
        obj.pAttnFlash64 = make_pipeline(dev, lib, @"qw3_vis_attn_flash64", err, errlen);
        obj.pResScale    = make_pipeline(dev, lib, @"qw3_vis_residual_scale", err, errlen);
        obj.pPatchMerge  = make_pipeline(dev, lib, @"qw3_vis_patch_merge", err, errlen);
        if (!obj.pPatchEmbed || !obj.pLayerNorm || !obj.pGemmF16 || !obj.pGemmGeneric ||
            !obj.pRope2D || !obj.pAttnFlash64 || !obj.pResScale || !obj.pPatchMerge) {
            qw3_vision_close(v);
            return -1;
        }

        v->obj = (__bridge_retained void *)obj;
        *out = v;
        return 0;
    }
}

void qw3_vision_close(qw3_vision *v) {
    if (!v) return;
    @autoreleasepool {
        if (v->obj) {
            (void)(__bridge_transfer QW3VisionObj *)v->obj;
            v->obj = NULL;
        }
    }
    free(v->layers);
    if (v->mmap_ptr && v->mmap_ptr != MAP_FAILED) {
        munmap(v->mmap_ptr, v->mmap_len);
    }
    if (v->fd >= 0) close(v->fd);
    free(v);
}

const char *qw3_vision_arch(const qw3_vision *v) {
    return v ? v->projector_type : "";
}

int qw3_vision_embd_dim(const qw3_vision *v) {
    return v ? v->proj_dim : 0;
}

int qw3_vision_max_tokens(const qw3_vision *v) {
    return v ? v->max_tokens : 0;
}

void qw3_vision_result_free(qw3_vision_result *r) {
    if (!r) return;
    free(r->embd);
    memset(r, 0, sizeof(*r));
}

/* ============================================================================
 * Image resize & Inkling2v preprocessing helpers
 * ============================================================================ */

static int align_up_int(int x, int a) {
    return ((x + a - 1) / a) * a;
}

/* Compute aspect-ratio-preserving target size bounded by max_pixels (aligned to align). */
static void calc_bounded_size(int src_w, int src_h, int align, int64_t max_pixels,
                              int *out_w, int *out_h) {
    int64_t aw = align_up_int(src_w, align);
    int64_t ah = align_up_int(src_h, align);
    if (max_pixels <= 0 || aw * ah <= max_pixels) {
        *out_w = src_w;
        *out_h = src_h;
        return;
    }
    double scale = sqrt((double)max_pixels / ((double)src_w * (double)src_h));
    int nw = (int)floor((double)src_w * scale);
    int nh = (int)floor((double)src_h * scale);
    nw = (nw / align) * align;
    nh = (nh / align) * align;
    if (nw < align) nw = align;
    if (nh < align) nh = align;
    while ((int64_t)nw * (int64_t)nh > max_pixels) {
        if (nw >= nh && nw > align) nw -= align;
        else if (nh > align) nh -= align;
        else break;
    }
    *out_w = nw;
    *out_h = nh;
}

/* Bilinear resize for raw RGB8 buffer (used when qw3_vision_encode_rgb gets huge images). */
static uint8_t *resize_rgb_bilinear(const uint8_t *src, int sw, int sh, int dw, int dh) {
    uint8_t *dst = (uint8_t *)malloc((size_t)dw * (size_t)dh * 3);
    if (!dst) return NULL;
    float sf_x = (float)dw / (float)sw;
    float sf_y = (float)dh / (float)sh;
    for (int y = 0; y < dh; y++) {
        float fy = ((float)y + 0.5f) / sf_y - 0.5f;
        int y0 = (int)floorf(fy), y1 = y0 + 1;
        if (y0 < 0) y0 = 0;
        if (y0 >= sh) y0 = sh - 1;
        if (y1 < 0) y1 = 0;
        if (y1 >= sh) y1 = sh - 1;
        float dy = fy - (float)y0;
        if (dy < 0.0f) dy = 0.0f;
        if (dy > 1.0f) dy = 1.0f;
        for (int x = 0; x < dw; x++) {
            float fx = ((float)x + 0.5f) / sf_x - 0.5f;
            int x0 = (int)floorf(fx), x1 = x0 + 1;
            if (x0 < 0) x0 = 0;
            if (x0 >= sw) x0 = sw - 1;
            if (x1 < 0) x1 = 0;
            if (x1 >= sw) x1 = sw - 1;
            float dx = fx - (float)x0;
            if (dx < 0.0f) dx = 0.0f;
            if (dx > 1.0f) dx = 1.0f;
            const uint8_t *p00 = src + ((size_t)y0 * sw + x0) * 3;
            const uint8_t *p10 = src + ((size_t)y0 * sw + x1) * 3;
            const uint8_t *p01 = src + ((size_t)y1 * sw + x0) * 3;
            const uint8_t *p11 = src + ((size_t)y1 * sw + x1) * 3;
            uint8_t *out = dst + ((size_t)y * dw + x) * 3;
            for (int c = 0; c < 3; c++) {
                float val = p00[c] * (1.0f - dx) * (1.0f - dy)
                          + p10[c] * dx * (1.0f - dy)
                          + p01[c] * (1.0f - dx) * dy
                          + p11[c] * dx * dy;
                int iv = (int)lroundf(val);
                if (iv < 0) iv = 0;
                if (iv > 255) iv = 255;
                out[c] = (uint8_t)iv;
            }
        }
    }
    return dst;
}

/* ============================================================================
 * Metal forward pass for inkling2v
 * ============================================================================ */

static id<MTLBuffer> ensure_buf(id<MTLDevice> dev, id<MTLBuffer> cur,
                                NSUInteger bytes, MTLResourceOptions opts) {
    if (cur && cur.length >= bytes) return cur;
    return [dev newBufferWithLength:bytes options:opts];
}

static void encode_gemm(QW3VisionObj *obj, id<MTLComputeCommandEncoder> enc,
                        id<MTLBuffer> X, const vis_tensor *W, const vis_tensor *B,
                        bool has_bias, bool act_gelu, id<MTLBuffer> Y,
                        uint32_t M, uint32_t N_out, uint32_t K) {
    vis_gemm_args a = { M, N_out, K, W->type, has_bias ? 1u : 0u, act_gelu ? 1u : 0u };
    id<MTLComputePipelineState> pso = (W->type == GGML_T_F16) ? obj.pGemmF16 : obj.pGemmGeneric;
    [enc setComputePipelineState:pso];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:X offset:0 atIndex:1];
    [enc setBuffer:obj.weightsBuf offset:(NSUInteger)W->offset atIndex:2];
    [enc setBuffer:obj.weightsBuf offset:(NSUInteger)(has_bias ? B->offset : 0) atIndex:3];
    [enc setBuffer:Y offset:0 atIndex:4];
    uint32_t bm = (W->type == GGML_T_F16) ? 64u : 32u;
    MTLSize grid = MTLSizeMake((N_out + 63u) / 64u, (M + bm - 1u) / bm, 1);
    MTLSize tg   = MTLSizeMake(128, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
}

static void encode_ln(QW3VisionObj *obj, id<MTLComputeCommandEncoder> enc,
                      id<MTLBuffer> X, const vis_tensor *W, const vis_tensor *B,
                      id<MTLBuffer> Y, uint32_t N, uint32_t D, float eps) {
    vis_ln_args a = { N, D, eps };
    [enc setComputePipelineState:obj.pLayerNorm];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:X offset:0 atIndex:1];
    [enc setBuffer:obj.weightsBuf offset:(NSUInteger)W->offset atIndex:2];
    [enc setBuffer:obj.weightsBuf offset:(NSUInteger)B->offset atIndex:3];
    [enc setBuffer:Y offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

int qw3_vision_encode_rgb(qw3_vision *v, const uint8_t *rgb,
                          int width, int height,
                          qw3_vision_result *out, char *err, size_t errlen) {
    if (!v || !rgb || width <= 0 || height <= 0 || !out) {
        if (err && errlen) snprintf(err, errlen, "invalid vision encode arguments");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    double t0 = now_ms();

    const int align = v->patch_size * v->n_merge;
    const int64_t max_pixels = (int64_t)v->max_tokens * (int64_t)align * (int64_t)align;

    int rw = width, rh = height;
    calc_bounded_size(width, height, align, max_pixels, &rw, &rh);
    uint8_t *resized = NULL;
    const uint8_t *src = rgb;
    if (rw != width || rh != height) {
        resized = resize_rgb_bilinear(rgb, width, height, rw, rh);
        if (!resized) {
            if (err && errlen) snprintf(err, errlen, "out of memory resizing image");
            return -1;
        }
        src = resized;
    }

    const int pad_w = align_up_int(rw, align);
    const int pad_h = align_up_int(rh, align);
    const int W_p = pad_w / v->patch_size;
    const int H_p = pad_h / v->patch_size;
    const uint32_t n_patches = (uint32_t)(W_p * H_p);
    const uint32_t N = n_patches + 1u; /* +1 for CLS */
    const uint32_t D = (uint32_t)v->n_embd;
    const uint32_t FF = (uint32_t)v->n_ff;
    const uint32_t out_w = (uint32_t)(W_p / v->n_merge);
    const uint32_t out_h = (uint32_t)(H_p / v->n_merge);
    const uint32_t n_out = out_w * out_h;
    const uint32_t merge_dim = (uint32_t)(v->n_merge * v->n_merge * v->n_embd);

    @autoreleasepool {
        QW3VisionObj *obj = (__bridge QW3VisionObj *)v->obj;
        id<MTLDevice> dev = obj.device;

        NSUInteger img_bytes   = (NSUInteger)3u * (NSUInteger)pad_w * (NSUInteger)pad_h * sizeof(float);
        NSUInteger seq_bytes   = (NSUInteger)N * (NSUInteger)D * sizeof(float);
        NSUInteger ff_bytes    = (NSUInteger)N * (NSUInteger)FF * sizeof(float);
        NSUInteger merge_bytes = (NSUInteger)n_out * (NSUInteger)merge_dim * sizeof(float);
        NSUInteger p1_bytes    = (NSUInteger)n_out * (NSUInteger)v->proj_hidden * sizeof(float);
        NSUInteger out_bytes   = (NSUInteger)n_out * (NSUInteger)v->proj_dim * sizeof(float);

        obj.bufImg   = ensure_buf(dev, obj.bufImg,   img_bytes,   MTLResourceStorageModeShared);
        obj.bufX     = ensure_buf(dev, obj.bufX,     seq_bytes,   MTLResourceStorageModePrivate);
        obj.bufH     = ensure_buf(dev, obj.bufH,     seq_bytes,   MTLResourceStorageModePrivate);
        obj.bufQ     = ensure_buf(dev, obj.bufQ,     seq_bytes,   MTLResourceStorageModePrivate);
        obj.bufK     = ensure_buf(dev, obj.bufK,     seq_bytes,   MTLResourceStorageModePrivate);
        obj.bufV     = ensure_buf(dev, obj.bufV,     seq_bytes,   MTLResourceStorageModePrivate);
        obj.bufAttn  = ensure_buf(dev, obj.bufAttn,  seq_bytes,   MTLResourceStorageModePrivate);
        obj.bufFF    = ensure_buf(dev, obj.bufFF,    ff_bytes,    MTLResourceStorageModePrivate);
        obj.bufMerge = ensure_buf(dev, obj.bufMerge, merge_bytes, MTLResourceStorageModePrivate);
        obj.bufP1    = ensure_buf(dev, obj.bufP1,    p1_bytes,    MTLResourceStorageModePrivate);
        obj.bufOut   = ensure_buf(dev, obj.bufOut,   out_bytes,   MTLResourceStorageModeShared);

        if (!obj.bufImg || !obj.bufX || !obj.bufH || !obj.bufQ || !obj.bufK ||
            !obj.bufV || !obj.bufAttn || !obj.bufFF || !obj.bufMerge ||
            !obj.bufP1 || !obj.bufOut) {
            free(resized);
            if (err && errlen) snprintf(err, errlen, "Metal vision buffer allocation failed");
            return -1;
        }

        /* Fill planar float32 normalized image [3, pad_h, pad_w].
         * Matches mtmd_image_preprocessor_inkling2v: pad value is -1.0f BEFORE
         * normalizing by (px - mean*255) / (std*255). */
        float *dst_img = (float *)obj.bufImg.contents;
        const size_t plane_sz = (size_t)pad_w * (size_t)pad_h;
        float inv_std[3], mean255[3], pad_val[3];
        for (int c = 0; c < 3; c++) {
            mean255[c] = v->image_mean[c] * 255.0f;
            inv_std[c] = 1.0f / (v->image_std[c] * 255.0f);
            pad_val[c] = (-1.0f - mean255[c]) * inv_std[c];
        }
        for (int c = 0; c < 3; c++) {
            float *plane = dst_img + (size_t)c * plane_sz;
            const float m = mean255[c], is = inv_std[c], pv = pad_val[c];
            for (int y = 0; y < rh; y++) {
                float *row = plane + (size_t)y * (size_t)pad_w;
                const uint8_t *srow = src + (size_t)y * (size_t)rw * 3u;
                for (int x = 0; x < rw; x++) {
                    row[x] = ((float)srow[x * 3 + c] - m) * is;
                }
                for (int x = rw; x < pad_w; x++) row[x] = pv;
            }
            for (int y = rh; y < pad_h; y++) {
                float *row = plane + (size_t)y * (size_t)pad_w;
                for (int x = 0; x < pad_w; x++) row[x] = pv;
            }
        }
        free(resized);

        id<MTLCommandBuffer> cb = [obj.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) {
            if (err && errlen) snprintf(err, errlen, "failed to create Metal command encoder for vision");
            return -1;
        }

        /* 1. Patch embedding + bilinear pos_embd + CLS prepend -> bufH */
        {
            vis_patch_args pa = {
                (uint32_t)pad_w, (uint32_t)pad_h, (uint32_t)v->patch_size,
                (uint32_t)W_p, (uint32_t)H_p, D, (uint32_t)v->pos_grid,
                v->patch_w.type, v->has_patch_b ? 1u : 0u
            };
            [enc setComputePipelineState:obj.pPatchEmbed];
            [enc setBytes:&pa length:sizeof(pa) atIndex:0];
            [enc setBuffer:obj.bufImg offset:0 atIndex:1];
            [enc setBuffer:obj.weightsBuf offset:(NSUInteger)v->patch_w.offset atIndex:2];
            [enc setBuffer:obj.weightsBuf offset:(NSUInteger)(v->has_patch_b ? v->patch_b.offset : 0) atIndex:3];
            [enc setBuffer:obj.weightsBuf offset:(NSUInteger)v->pos_embd.offset atIndex:4];
            [enc setBuffer:obj.weightsBuf offset:(NSUInteger)v->class_embd.offset atIndex:5];
            [enc setBuffer:obj.weightsBuf offset:(NSUInteger)v->class_pos_embd.offset atIndex:6];
            [enc setBuffer:obj.bufH offset:0 atIndex:7];
            [enc dispatchThreads:MTLSizeMake(D, N, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
        }

        /* 2. Pre-LayerNorm: bufH -> bufX */
        encode_ln(obj, enc, obj.bufH, &v->pre_ln_w, &v->pre_ln_b, obj.bufX, N, D, v->eps);

        /* 3. 23 ViT Transformer blocks */
        const float attn_scale = 1.0f / sqrtf((float)v->d_head);
        for (int il = 0; il < v->n_layer; il++) {
            const vis_layer *L = &v->layers[il];

            /* LN1: bufX -> bufH */
            encode_ln(obj, enc, obj.bufX, &L->ln1_w, &L->ln1_b, obj.bufH, N, D, v->eps);

            /* Q, K, V projections */
            encode_gemm(obj, enc, obj.bufH, &L->q_w, &L->q_b, true, false, obj.bufQ, N, D, D);
            encode_gemm(obj, enc, obj.bufH, &L->k_w, &L->k_b, true, false, obj.bufK, N, D, D);
            encode_gemm(obj, enc, obj.bufH, &L->v_w, &L->v_b, true, false, obj.bufV, N, D, D);

            /* 2D RoPE on Q and K */
            {
                vis_rope_args ra = { N, (uint32_t)W_p, (uint32_t)v->n_head, (uint32_t)v->d_head, v->rope_theta };
                [enc setComputePipelineState:obj.pRope2D];
                [enc setBytes:&ra length:sizeof(ra) atIndex:0];
                [enc setBuffer:obj.bufQ offset:0 atIndex:1];
                [enc setBuffer:obj.bufK offset:0 atIndex:2];
                uint32_t pairs = (uint32_t)(v->n_head * (v->d_head / 2));
                [enc dispatchThreads:MTLSizeMake(pairs, N, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
            }

            /* Bidirectional FlashAttention: Q, K, V -> bufAttn */
            {
                vis_attn_args aa = { N, (uint32_t)v->n_head, (uint32_t)v->d_head, attn_scale };
                [enc setComputePipelineState:obj.pAttnFlash64];
                [enc setBytes:&aa length:sizeof(aa) atIndex:0];
                [enc setBuffer:obj.bufQ offset:0 atIndex:1];
                [enc setBuffer:obj.bufK offset:0 atIndex:2];
                [enc setBuffer:obj.bufV offset:0 atIndex:3];
                [enc setBuffer:obj.bufAttn offset:0 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((N + 31u) / 32u, (uint32_t)v->n_head, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }

            /* Attention output projection: bufAttn -> bufH */
            encode_gemm(obj, enc, obj.bufAttn, &L->o_w, &L->o_b, true, false, obj.bufH, N, D, D);

            /* Residual + LayerScale 1: bufX += bufH * ls1 */
            {
                vis_res_args ra = { N, D };
                [enc setComputePipelineState:obj.pResScale];
                [enc setBytes:&ra length:sizeof(ra) atIndex:0];
                [enc setBuffer:obj.bufX offset:0 atIndex:1];
                [enc setBuffer:obj.bufH offset:0 atIndex:2];
                [enc setBuffer:obj.weightsBuf offset:(NSUInteger)L->ls1_w.offset atIndex:3];
                [enc dispatchThreads:MTLSizeMake(D, N, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
            }

            /* LN2: bufX -> bufH */
            encode_ln(obj, enc, obj.bufX, &L->ln2_w, &L->ln2_b, obj.bufH, N, D, v->eps);

            /* FFN up + GELU_ERF: bufH -> bufFF */
            encode_gemm(obj, enc, obj.bufH, &L->ff_up_w, &L->ff_up_b, true, true, obj.bufFF, N, FF, D);

            /* FFN down: bufFF -> bufH */
            encode_gemm(obj, enc, obj.bufFF, &L->ff_down_w, &L->ff_down_b, true, false, obj.bufH, N, D, FF);

            /* Residual + LayerScale 2: bufX += bufH * ls2 */
            {
                vis_res_args ra = { N, D };
                [enc setComputePipelineState:obj.pResScale];
                [enc setBytes:&ra length:sizeof(ra) atIndex:0];
                [enc setBuffer:obj.bufX offset:0 atIndex:1];
                [enc setBuffer:obj.bufH offset:0 atIndex:2];
                [enc setBuffer:obj.weightsBuf offset:(NSUInteger)L->ls2_w.offset atIndex:3];
                [enc dispatchThreads:MTLSizeMake(D, N, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
            }
        }

        /* 4. Drop CLS + 2x2 spatial patch merge: bufX -> bufMerge */
        {
            vis_merge_args ma = { (uint32_t)W_p, (uint32_t)H_p, D, (uint32_t)v->n_merge };
            [enc setComputePipelineState:obj.pPatchMerge];
            [enc setBytes:&ma length:sizeof(ma) atIndex:0];
            [enc setBuffer:obj.bufX offset:0 atIndex:1];
            [enc setBuffer:obj.bufMerge offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(merge_dim, n_out, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
        }

        /* 5. Projector MLP: bufMerge -> bufP1 (GELU_ERF) -> bufOut */
        encode_gemm(obj, enc, obj.bufMerge, &v->mm1_w, &v->mm1_b, v->has_mm1_b, true,
                    obj.bufP1, n_out, (uint32_t)v->proj_hidden, merge_dim);
        encode_gemm(obj, enc, obj.bufP1, &v->mm2_w, &v->mm2_b, v->has_mm2_b, false,
                    obj.bufOut, n_out, (uint32_t)v->proj_dim, (uint32_t)v->proj_hidden);

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status != MTLCommandBufferStatusCompleted) {
            if (err && errlen) {
                snprintf(err, errlen, "Metal vision execution failed: %s",
                         cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
            }
            return -1;
        }

        float *embd = (float *)malloc(out_bytes);
        if (!embd) {
            if (err && errlen) snprintf(err, errlen, "out of memory allocating vision output");
            return -1;
        }
        memcpy(embd, obj.bufOut.contents, out_bytes);

        out->embd = embd;
        out->n_tokens = (int)n_out;
        out->embd_dim = v->proj_dim;
        out->grid_w = (int)out_w;
        out->grid_h = (int)out_h;
        out->proc_w = pad_w;
        out->proc_h = pad_h;
        out->encode_ms = now_ms() - t0;
        return 0;
    }
}

/* ============================================================================
 * ImageIO decode (PNG, JPEG, WebP, GIF, HEIC, BMP, TIFF)
 * ============================================================================ */

int qw3_vision_encode_bytes(qw3_vision *v, const uint8_t *bytes, size_t len,
                            qw3_vision_result *out, char *err, size_t errlen) {
    if (!v || !bytes || len == 0 || !out) {
        if (err && errlen) snprintf(err, errlen, "empty image payload");
        return -1;
    }
    @autoreleasepool {
        CFDataRef cfdata = CFDataCreateWithBytesNoCopy(kCFAllocatorDefault, bytes, (CFIndex)len, kCFAllocatorNull);
        if (!cfdata) {
            if (err && errlen) snprintf(err, errlen, "failed to wrap image bytes");
            return -1;
        }
        CGImageSourceRef src = CGImageSourceCreateWithData(cfdata, NULL);
        CFRelease(cfdata);
        if (!src) {
            if (err && errlen) snprintf(err, errlen, "unsupported or invalid image data");
            return -1;
        }
        CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
        CFRelease(src);
        if (!img) {
            if (err && errlen) snprintf(err, errlen, "failed to decode image");
            return -1;
        }

        int src_w = (int)CGImageGetWidth(img);
        int src_h = (int)CGImageGetHeight(img);
        if (src_w <= 0 || src_h <= 0) {
            CGImageRelease(img);
            if (err && errlen) snprintf(err, errlen, "decoded image has zero dimensions");
            return -1;
        }

        /* Resize during CoreGraphics rasterization if larger than max_pixels (high quality). */
        const int align = v->patch_size * v->n_merge;
        const int64_t max_pixels = (int64_t)v->max_tokens * (int64_t)align * (int64_t)align;
        int dst_w = src_w, dst_h = src_h;
        calc_bounded_size(src_w, src_h, align, max_pixels, &dst_w, &dst_h);

        size_t rgba_len = (size_t)dst_w * (size_t)dst_h * 4u;
        uint8_t *rgba = (uint8_t *)malloc(rgba_len);
        uint8_t *rgb  = (uint8_t *)malloc((size_t)dst_w * (size_t)dst_h * 3u);
        if (!rgba || !rgb) {
            free(rgba);
            free(rgb);
            CGImageRelease(img);
            if (err && errlen) snprintf(err, errlen, "out of memory decoding image (%dx%d)", dst_w, dst_h);
            return -1;
        }

        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        if (!cs) cs = CGColorSpaceCreateDeviceRGB();
        CGContextRef ctx = CGBitmapContextCreate(
            rgba, (size_t)dst_w, (size_t)dst_h, 8, (size_t)dst_w * 4u, cs,
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        CGColorSpaceRelease(cs);
        if (!ctx) {
            free(rgba);
            free(rgb);
            CGImageRelease(img);
            if (err && errlen) snprintf(err, errlen, "failed to create bitmap context");
            return -1;
        }
        /* Composite transparent images cleanly over white background */
        CGContextSetRGBFillColor(ctx, 1.0, 1.0, 1.0, 1.0);
        CGContextFillRect(ctx, CGRectMake(0, 0, dst_w, dst_h));
        CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
        CGContextDrawImage(ctx, CGRectMake(0, 0, dst_w, dst_h), img);
        CGContextRelease(ctx);
        CGImageRelease(img);

        size_t npix = (size_t)dst_w * (size_t)dst_h;
        for (size_t i = 0; i < npix; i++) {
            rgb[i * 3 + 0] = rgba[i * 4 + 0];
            rgb[i * 3 + 1] = rgba[i * 4 + 1];
            rgb[i * 3 + 2] = rgba[i * 4 + 2];
        }
        free(rgba);

        int rc = qw3_vision_encode_rgb(v, rgb, dst_w, dst_h, out, err, errlen);
        free(rgb);
        return rc;
    }
}

int qw3_vision_encode_file(qw3_vision *v, const char *path,
                           qw3_vision_result *out, char *err, size_t errlen) {
    if (!path || !path[0]) {
        if (err && errlen) snprintf(err, errlen, "empty image path");
        return -1;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        if (err && errlen) snprintf(err, errlen, "cannot open image '%s': %s", path, strerror(errno));
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return -1; }
    long sz = ftell(fp);
    if (sz <= 0 || sz > 128L * 1024L * 1024L) {
        fclose(fp);
        if (err && errlen) snprintf(err, errlen, "invalid image file size (%ld bytes)", sz);
        return -1;
    }
    rewind(fp);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fclose(fp); return -1; }
    size_t nr = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (nr != (size_t)sz) {
        free(buf);
        if (err && errlen) snprintf(err, errlen, "short read on '%s'", path);
        return -1;
    }
    int rc = qw3_vision_encode_bytes(v, buf, (size_t)sz, out, err, errlen);
    free(buf);
    return rc;
}
