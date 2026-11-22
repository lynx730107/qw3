#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
int main(void) {
    @autoreleasepool {
        id<MTLDevice> d=MTLCreateSystemDefaultDevice();
        if(!d) return 1;
        printf("%s\n",d.name.UTF8String);
        for(id<MTLCounterSet> set in d.counterSets){
            printf("set %s\n",set.name.UTF8String);
            for(id<MTLCounter> c in set.counters) printf("  %s\n",c.name.UTF8String);
        }
        printf("stage=%d dispatch=%d\n",[d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary],
               [d supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]);
    }
}
