#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(void) { @autoreleasepool {
 id<MTLDevice> d=MTLCreateSystemDefaultDevice();
 for (int i=0;i<2;i++) {
  id<MTLBuffer> b=[d newBufferWithLength:4096 options:MTLResourceStorageModePrivate | (i ? MTLResourceHazardTrackingModeUntracked : 0)];
  printf("device=%s requested_untracked=%d actual_mode=%lu\n",d.name.UTF8String,i,(unsigned long)b.hazardTrackingMode);
 }
} }
