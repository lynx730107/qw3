#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
static void probe_emit(id obj) {
    NSData *j=[NSJSONSerialization dataWithJSONObject:obj options:NSJSONWritingPrettyPrinted error:nil];
    printf("%s\n",[[NSString alloc] initWithData:j encoding:NSUTF8StringEncoding].UTF8String);fflush(stdout);
}
int main(void){ @autoreleasepool {id<MTLDevice> g_device=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> g_queue=[g_device newCommandQueue];
            NSString *source=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void init(device uint4 *p [[buffer(0)]],uint i [[thread_position_in_grid]]) {p[i]=uint4(i,i+1,i+2,i+3)^uint4(0x9e3779b9u);}\nkernel void stream(device const uint4 *p [[buffer(0)]],device uint *o [[buffer(1)]],constant uint &n [[buffer(2)]],uint i [[thread_position_in_grid]],uint stride [[threads_per_grid]]) {uint4 s=uint4(0);for(uint j=i;j<n;j+=stride) s+=p[j];o[i]=s.x^s.y^s.z^s.w;}";
            NSError *err=nil;id<MTLLibrary> lib=[g_device newLibraryWithSource:source options:nil error:&err];
            id<MTLComputePipelineState> init=[g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"init"] error:&err];
            id<MTLComputePipelineState> stream=[g_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stream"] error:&err];
            if (!init || !stream) { fprintf(stderr,"stream compile %s\n",err.localizedDescription.UTF8String);return 1; }
            // Mapped model pages stay untouched in this process. No inference overlap.
            NSMutableArray *runs=[NSMutableArray new];
            for (int mb=128;mb<=1024;mb*=2) {
                uint64_t bytes=(uint64_t)mb*1024*1024;uint32_t n=(uint32_t)(bytes/16);
                uint32_t threads=1024*256;
                id<MTLBuffer> buf=[g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                id<MTLBuffer> out=[g_device newBufferWithLength:threads*4 options:MTLResourceStorageModeShared];
                if (!buf || !out) return 1;
                id<MTLCommandBuffer> cb=[g_queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                [enc setComputePipelineState:init];[enc setBuffer:buf offset:0 atIndex:0];
                [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
                NSMutableArray *samples=[NSMutableArray new];uint32_t checksum=0;
                for (int rep=0;rep<9;rep++) {
                    cb=[g_queue commandBuffer];enc=[cb computeCommandEncoder];
                    [enc setComputePipelineState:stream];[enc setBuffer:buf offset:0 atIndex:0];[enc setBuffer:out offset:0 atIndex:1];[enc setBytes:&n length:4 atIndex:2];
                    for (int read=0;read<8;read++) {
                        [enc dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
                    if (cb.status!=MTLCommandBufferStatusCompleted) return 1;
                    if (rep>=3) [samples addObject:@((cb.GPUEndTime-cb.GPUStartTime)*1000/8)];
                    uint32_t *p=out.contents;
                    const uint32_t lanes[]={0u,1u,17u,123u,threads/2,threads-1};
                    for (unsigned lane=0;lane<sizeof(lanes)/sizeof(lanes[0]);lane++) {
                        uint32_t sum[4]={0};
                        for (uint32_t j=lanes[lane];j<n;j+=threads)
                            for (uint32_t k=0;k<4;k++) sum[k]+=(j+k)^0x9e3779b9u;
                        if (p[lanes[lane]]!=(sum[0]^sum[1]^sum[2]^sum[3])) {
                            fprintf(stderr,"stream oracle mismatch size=%d lane=%u\n",mb,lanes[lane]);return 1;
                        }
                    }
                    uint32_t check=0;for(uint32_t i=0;i<threads;i++) check^=p[i];
                    if (rep && check!=checksum) return 1;checksum=check;
                }
                [runs addObject:@{@"bytes":@(bytes),@"gpu_ms":samples,@"checksum":@(checksum),@"reads_per_command":@8}];
            }
            probe_emit(@{@"scope":@"read-only GPU streaming, eight ordered reads per command, three warmups, six timed samples; includes barriers, NOT inference DRAM traffic",@"runs":runs});
        }
return 0;}
