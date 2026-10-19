#pragma once
#import <Metal/Metal.h>

// Query the real runtime capability when building with an older Apple SDK.
@protocol MVKPlacementSparseCapability
- (BOOL)supportsPlacementSparse;
@end

static inline bool mvkSupportsPlacementSparse(id<MTLDevice> device) {
    return [device respondsToSelector:@selector(supportsPlacementSparse)] &&
           [(id<MVKPlacementSparseCapability>)device supportsPlacementSparse];
}
