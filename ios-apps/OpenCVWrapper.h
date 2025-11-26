// OpenCVWrapper.h
// Add to your target and import via Bridging Header.

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_SWIFT_NAME(OpenCVWrapper)
@interface OpenCVWrapper : NSObject

/// Tenengrad (Sobel-based) sharpness on entire image.
/// @param image input UIImage
/// @param ksize Sobel kernel size (1, 3, 5)
/// @param maxLongEdge downscale so max(width,height) <= this (0 = no downscale)
/// Tenengrad (Sobel-based) sharpness **inside mask** only (mask should be 0/255 grayscale).
+ (double)tenengradFor:(UIImage *)image
                  mask:(UIImage *)mask
                kernel:(NSInteger)ksize
           maxLongEdge:(NSInteger)maxLongEdge;

/// Laplacian variance **inside mask** only (mask should be 0/255 grayscale).
+ (double)laplacianVarianceFor:(UIImage *)image
                          mask:(UIImage *)mask;

@end
