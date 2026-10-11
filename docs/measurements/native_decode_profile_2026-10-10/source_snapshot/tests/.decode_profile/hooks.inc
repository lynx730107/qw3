/* Private experiments: no changes to production command boundaries or shaders. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
static int probe_capture;
static id<MTLCounterSampleBuffer> probe_samples;
static NSMutableArray<id<MTLCounterSampleBuffer>> *probe_sample_pages;
static NSMutableArray<NSMutableDictionary *> *probe_passes;
static NSMutableArray<id<MTLCommandBuffer>> *probe_commands;
static uint64_t probe_q3_calls;
static id<MTLComputePipelineState> probe_q3_base,probe_q3_candidate;
static id<MTLComputePipelineState> probe_iq3_base,probe_iq4_base,probe_iq3_candidate,probe_iq4_candidate;
static uint64_t probe_iq_calls, probe_iq3_calls, probe_iq4_calls;
static void *probe_pass_key=&probe_pass_key;
static void *probe_pipeline_key=&probe_pipeline_key;
@protocol QW3ProjectionHooks
- (id)probePipeline:(id<MTLFunction>)f error:(NSError **)error;
- (id)probeEncoder;
- (id)probeEncoderWithDispatchType:(MTLDispatchType)type;
- (void)probeSetPipeline:(id<MTLComputePipelineState>)pipeline;
- (void)probeDispatch:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;
- (void)probeThreads:(MTLSize)grid threadsPerThreadgroup:(MTLSize)threads;
- (void)probeCommit;
- (id)probeCommandQueue;
- (id)probeCommandBuffer;
- (id)probeCommandBufferUnretained;
@end
@interface NSObject (QW3ProjectionProbe) <QW3ProjectionHooks>
@end
static id<MTLComputeCommandEncoder> probe_encoder(id<MTLCommandBuffer> cb,MTLDispatchType type) {
    if (!probe_capture) return type==MTLDispatchTypeConcurrent ?
        [cb computeCommandEncoderWithDispatchType:type] : [cb computeCommandEncoder];
    if (probe_passes.count>=65536) abort();
    NSUInteger i=2*(probe_passes.count%2048);
    NSUInteger page=probe_passes.count/2048;
    MTLComputePassDescriptor *p=[MTLComputePassDescriptor computePassDescriptor];
    p.dispatchType=type;
    p.sampleBufferAttachments[0].sampleBuffer=probe_sample_pages[page];
    p.sampleBufferAttachments[0].startOfEncoderSampleIndex=i;
    p.sampleBufferAttachments[0].endOfEncoderSampleIndex=i+1;
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoderWithDescriptor:p];
    NSMutableDictionary *record=[@{@"sample_index":@(i),@"sample_page":@(page),@"dispatches":[NSMutableDictionary new]} mutableCopy];
    [probe_passes addObject:record];
    objc_setAssociatedObject(enc,probe_pass_key,record,OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    return enc;
}
static void probe_dispatch(id<MTLComputeCommandEncoder> enc) {
    if (!probe_capture) return;
    NSMutableDictionary *r=objc_getAssociatedObject(enc,probe_pass_key);
    NSString *name=objc_getAssociatedObject(enc,probe_pipeline_key);
    NSMutableDictionary *d=r[@"dispatches"];
    if (!r || !name) abort();
    d[name]=@([d[name] unsignedLongLongValue]+1);
}
@implementation NSObject (QW3ProjectionProbe)
- (id)probePipeline:(id<MTLFunction>)f error:(NSError **)error {
    id<MTLComputePipelineState> p=[(id<MTLDevice>)self newComputePipelineStateWithFunction:f error:error];
    if (p) objc_setAssociatedObject(p,probe_pipeline_key,f.name,OBJC_ASSOCIATION_RETAIN_NONATOMIC); return p;
}
- (id)probeEncoder { return probe_encoder((id)self,MTLDispatchTypeSerial); }
- (id)probeEncoderWithDispatchType:(MTLDispatchType)type { return probe_encoder((id)self,type); }
- (void)probeSetPipeline:(id<MTLComputePipelineState>)p {
    if (p==probe_q3_candidate) probe_q3_calls++;
    if (p==probe_iq3_candidate || p==probe_iq4_candidate) probe_iq_calls++;
    if (p==probe_iq3_candidate) probe_iq3_calls++;
    if (p==probe_iq4_candidate) probe_iq4_calls++;
    [(id<MTLComputeCommandEncoder>)self setComputePipelineState:p];
    if (probe_capture) objc_setAssociatedObject(self,probe_pipeline_key,objc_getAssociatedObject(p,probe_pipeline_key),OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}
- (void)probeDispatch:(MTLSize)g threadsPerThreadgroup:(MTLSize)t {
    [(id<MTLComputeCommandEncoder>)self dispatchThreadgroups:g threadsPerThreadgroup:t]; probe_dispatch((id)self);
}
- (void)probeThreads:(MTLSize)g threadsPerThreadgroup:(MTLSize)t {
    [(id<MTLComputeCommandEncoder>)self dispatchThreads:g threadsPerThreadgroup:t]; probe_dispatch((id)self);
}
- (id)probeCommandQueue { return [(id<MTLDevice>)self newCommandQueue]; }
- (id)probeCommandBuffer { return [(id<MTLCommandQueue>)self commandBuffer]; }
- (id)probeCommandBufferUnretained { return [(id<MTLCommandQueue>)self commandBufferWithUnretainedReferences]; }
- (void)probeCommit {
    if (probe_capture) [probe_commands addObject:(id)self];
    [(id<MTLCommandBuffer>)self commit];
}
@end
static id probe_default_device(void) { return MTLCreateSystemDefaultDevice(); }
static __attribute__((unused)) id probe_all_devices(void) { return MTLCopyAllDevices(); }
#define MTLCreateSystemDefaultDevice probe_default_device
#define MTLCopyAllDevices probe_all_devices
#define MTLCommandQueue MTLCommandQueue,QW3ProjectionHooks
#define newCommandQueue probeCommandQueue
#define commandBuffer probeCommandBuffer
#define commandBufferWithUnretainedReferences probeCommandBufferUnretained
#define MTLDevice MTLDevice,QW3ProjectionHooks
#define MTLCommandBuffer MTLCommandBuffer,QW3ProjectionHooks
#define MTLComputeCommandEncoder MTLComputeCommandEncoder,QW3ProjectionHooks
#define newComputePipelineStateWithFunction probePipeline
#define computeCommandEncoder probeEncoder
#define computeCommandEncoderWithDispatchType probeEncoderWithDispatchType
#define setComputePipelineState probeSetPipeline
#define dispatchThreadgroups probeDispatch
#define dispatchThreads probeThreads
#define commit probeCommit
#define qw3_metal_begin_commands profile_native_begin_commands
#include "../../qw3_metal.m"
#undef qw3_metal_begin_commands
int qw3_metal_begin_commands(void) {
    int ok=profile_native_begin_commands();
    if (ok && probe_capture) g_batch_separate_encoders=1;
    return ok;
}
#undef MTLCreateSystemDefaultDevice
#undef MTLCopyAllDevices
#undef MTLCommandQueue
#undef newCommandQueue
#undef commandBuffer
#undef commandBufferWithUnretainedReferences
#undef MTLDevice
#undef MTLCommandBuffer
#undef MTLComputeCommandEncoder
#undef newComputePipelineStateWithFunction
#undef computeCommandEncoder
#undef computeCommandEncoderWithDispatchType
#undef setComputePipelineState
#undef dispatchThreadgroups
#undef dispatchThreads
#undef commit

static __attribute__((unused)) int probe_profile_initialize(void) {
    if (probe_samples) return 1;
    if (![g_device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) return 0;
    MTLCounterSampleBufferDescriptor *d=[MTLCounterSampleBufferDescriptor new];
    for (id<MTLCounterSet> set in g_device.counterSets) if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) d.counterSet=set;
    if (!d.counterSet) { fprintf(stderr,"timestamp set missing wanted=%s sets=%s\n",MTLCommonCounterSetTimestamp.UTF8String,g_device.counterSets.description.UTF8String); return 0; }
    d.sampleCount=4096; d.storageMode=MTLStorageModeShared;
    NSError *err=nil;probe_sample_pages=[NSMutableArray new];
    for (int page=0;page<32;page++) {
        id<MTLCounterSampleBuffer> b=[g_device newCounterSampleBufferWithDescriptor:d error:&err];
        if (!b) { fprintf(stderr,"counter buffer unavailable: %s\n",err.localizedDescription.UTF8String);return 0; }
        [probe_sample_pages addObject:b];
    }
    probe_samples=probe_sample_pages[0];
    if (!probe_samples) fprintf(stderr,"counter buffer unavailable: %s\n",err.localizedDescription.UTF8String);
    return probe_samples!=nil;
}
static __attribute__((unused)) void probe_profile_start(void) {
    probe_passes=[NSMutableArray new];probe_commands=[NSMutableArray new];probe_capture=1;
}
static __attribute__((unused)) void probe_profile_end(int rep,double wall) {
    probe_capture=0;
    double total=0;
    for (NSUInteger page=0;page*2048<probe_passes.count;page++) {
        NSUInteger passes=MIN((NSUInteger)2048,probe_passes.count-page*2048);
        NSData *data=[probe_sample_pages[page] resolveCounterRange:NSMakeRange(0,2*passes)];
        if (data.length!=2*passes*sizeof(MTLCounterResultTimestamp)) abort();
        const MTLCounterResultTimestamp *t=data.bytes;
        for (NSUInteger pass=0;pass<passes;pass++) {
            NSMutableDictionary *p=probe_passes[page*2048+pass];NSUInteger i=2*pass;
            if (t[i].timestamp==MTLCounterErrorValue || t[i+1].timestamp==MTLCounterErrorValue || t[i+1].timestamp<t[i].timestamp) abort();
            p[@"gpu_ms"]=@((t[i+1].timestamp-t[i].timestamp)*1e-6);
        }
    }
    for (id<MTLCommandBuffer> cb in probe_commands) {
        if (cb.status!=MTLCommandBufferStatusCompleted) abort();
        total+=(cb.GPUEndTime-cb.GPUStartTime)*1000;
    }
    NSDictionary *r=@{@"rep":@(rep),@"wall_ms":@(wall*1000),@"command_gpu_ms_sum":@(total),
        @"command_buffers":@(probe_commands.count),@"passes":probe_passes,
        @"dispatch_boundary_supported":@([g_device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary])};
    NSData *j=[NSJSONSerialization dataWithJSONObject:r options:0 error:nil];
    fprintf(stderr,"projection_profile %s\n",[[NSString alloc] initWithData:j encoding:NSUTF8StringEncoding].UTF8String);
    probe_passes=nil;probe_commands=nil;
}
