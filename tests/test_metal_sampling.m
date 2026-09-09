#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { float value; uint32_t id; } item;
static int compare(const void *a, const void *b) {
    const item *x = a, *y = b;
    if (x->value != y->value) return x->value > y->value ? -1 : 1;
    return (x->id > y->id) - (x->id < y->id);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *path = argc > 1 ? argv[1] : "metal/qw3_core_argmax.metal";
        NSUInteger threads = argc > 2 ? strtoul(argv[2], NULL, 10) : 256;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSString *body = [NSString stringWithContentsOfFile:@(path)
            encoding:NSUTF8StringEncoding error:&error];
        if (!device || !body) return 1;
        NSString *source = [@"#include <metal_stdlib>\nusing namespace metal;\n"
            stringByAppendingString:body];
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeFast;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        id<MTLFunction> fn = [library newFunctionWithName:@"qw3_topk_penalty_blocks"];
        id<MTLComputePipelineState> pipeline = fn ?
            [device newComputePipelineStateWithFunction:fn error:&error] : nil;
        if (!pipeline) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        const uint32_t capacity = 248320;
        id<MTLBuffer> input = [device newBufferWithLength:capacity * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> seen = [device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
        id<MTLBuffer> values = [device newBufferWithLength:((capacity+255)/256)*64*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> ids = [device newBufferWithLength:values.length options:MTLResourceStorageModeShared];
        const uint32_t sizes[] = {1, 17, 255, 256, 257, capacity};
        const uint32_t ks[] = {1, 20, 40, 64};
        for (unsigned mode = 0; mode < 3; mode++) {
            uint32_t rng = 42;
            for (uint32_t i = 0; i < capacity; i++) {
                rng = rng * 1664525u + 1013904223u;
                ((float *)input.contents)[i] = mode == 2 ? -2.0f :
                    (float)((int)(rng % 2001) - 1000) / 32.0f;
                ((unsigned char *)seen.contents)[i] = i % 3 == 0;
            }
            for (unsigned si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
                for (unsigned ki = 0; ki < sizeof(ks)/sizeof(ks[0]); ki++) {
                    struct { uint32_t n, k; float penalty; uint32_t apply; } args =
                        {sizes[si], ks[ki], 2.0f, mode == 1};
                    uint32_t blocks = (args.n + 255)/256;
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pipeline];
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    [enc setBuffer:input offset:0 atIndex:1];
                    [enc setBuffer:seen offset:0 atIndex:2];
                    [enc setBuffer:values offset:0 atIndex:3];
                    [enc setBuffer:ids offset:0 atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(blocks,1,1)
                        threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    if (cb.status == MTLCommandBufferStatusError) return 1;
                    for (uint32_t block = 0; block < blocks; block++) {
                        item expected[256];
                        for (uint32_t i = 0; i < 256; i++) {
                            uint32_t id = block*256+i;
                            float v = id < args.n ? ((float *)input.contents)[id] : -FLT_MAX;
                            if (id < args.n && args.apply && ((unsigned char *)seen.contents)[id])
                                v = v < 0 ? v * args.penalty : v / args.penalty;
                            expected[i] = (item){v, id < args.n ? id : UINT32_MAX};
                        }
                        qsort(expected, 256, sizeof(item), compare);
                        for (uint32_t i = 0; i < args.k; i++) {
                            uint32_t off = block*args.k+i;
                            float actual = ((float *)values.contents)[off];
                            if (((uint32_t *)ids.contents)[off] != expected[i].id ||
                                !isfinite(actual) || fabsf(actual-expected[i].value) > 0.00001f) {
                                fprintf(stderr,"FAIL mode=%u n=%u k=%u block=%u rank=%u\n",mode,args.n,args.k,block,i);
                                return 1;
                            }
                        }
                    }
                    if (args.n == capacity && args.k == 20 && mode == 1)
                        printf("top-k GPU %.3f ms\n", (cb.GPUEndTime-cb.GPUStartTime)*1000);
                }
            }
        }
        puts("test-metal-sampling: ok (72 cases, CPU sorted reference)");
    }
    return 0;
}
