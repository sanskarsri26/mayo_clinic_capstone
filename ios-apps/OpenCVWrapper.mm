// OpenCVWrapper.mm
// Must be compiled as Objective-C++ (.mm)

#import "OpenCVWrapper.h"
#import <CoreGraphics/CoreGraphics.h>

// Only the specific OpenCV headers we need
#include <opencv2/core.hpp>
// rah
#include <opencv2/imgproc.hpp>

#include <algorithm>   // std::max
#include <cmath>       // std::round
#include <cstring>     // memcpy

#pragma mark - UIImage ⇄ cv::Mat helpers (namespace-safe)

static cv::Mat uiimageToMatRGBA(const UIImage *image) {
    CGImageRef cgImage = image.CGImage;
    if (!cgImage) { return cv::Mat(); }

    const size_t width  = CGImageGetWidth(cgImage);
    const size_t height = CGImageGetHeight(cgImage);

    cv::Mat rgba((int)height, (int)width, CV_8UC4);

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGBitmapInfo bitmapInfo = kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault;

    CGContextRef ctx = CGBitmapContextCreate(rgba.data,
                                             (int)width,
                                             (int)height,
                                             8,
                                             (int)rgba.step[0],
                                             colorSpace,
                                             bitmapInfo);
    if (ctx) {
        CGContextDrawImage(ctx, CGRectMake(0, 0, width, height), cgImage);
        CGContextRelease(ctx);
    }
    if (colorSpace) { CGColorSpaceRelease(colorSpace); }
    return rgba;
}

static cv::Mat uiimageToMatGray(const UIImage *image) {
    cv::Mat rgba = uiimageToMatRGBA(image);
    if (rgba.empty()) { return cv::Mat(); }
    cv::Mat gray;
    cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
    return gray;
}

static void resizeKeepingLongEdge(const cv::Mat &src, cv::Mat &dst, int maxLongEdge) {
    if (maxLongEdge <= 0) { dst = src; return; }
    const int h = src.rows, w = src.cols;
    const int longEdge = std::max(h, w);
    if (longEdge <= maxLongEdge) { dst = src; return; }
    const double scale = static_cast<double>(maxLongEdge) / static_cast<double>(longEdge);
    cv::Size newSize((int)std::round(w * scale), (int)std::round(h * scale));
    cv::resize(src, dst, newSize, 0, 0, cv::INTER_AREA);
}

/// Ensure mask is binary (0/255) and matches target size (nearest).
static cv::Mat uiimageMaskToMatBinary(const UIImage *maskImage, const cv::Size &targetSize) {
    cv::Mat m = uiimageToMatGray(maskImage);
    if (m.empty()) { return cv::Mat(); }
    if (m.size() != targetSize) {
        cv::Mat r; cv::resize(m, r, targetSize, 0, 0, cv::INTER_NEAREST);
        m = r;
    }
    cv::threshold(m, m, 127, 255, cv::THRESH_BINARY);
    return m;
}

/// Mean of single-channel image (float or 8U), optional mask.
static double maskedMean(const cv::Mat &src, const cv::Mat &mask = cv::Mat()) {
    cv::Mat f;
    if (src.type() != CV_32F) { src.convertTo(f, CV_32F); } else { f = src; }
    cv::Scalar m = mask.empty() ? cv::mean(f) : cv::mean(f, mask);
    return m[0];
}

/// Variance (stddev^2), optional mask.
static double maskedVariance(const cv::Mat &src, const cv::Mat &mask = cv::Mat()) {
    cv::Mat f;
    if (src.type() != CV_32F) { src.convertTo(f, CV_32F); } else { f = src; }
    cv::Scalar mean, stddev;
    if (mask.empty()) {
        cv::meanStdDev(f, mean, stddev);
    } else {
        cv::meanStdDev(f, mean, stddev, mask);
    }
    return stddev[0] * stddev[0];
}

#pragma mark - Implementation

@implementation OpenCVWrapper

+ (double)tenengradFor:(UIImage *)image mask:(UIImage *)mask kernel:(NSInteger)ksize maxLongEdge:(NSInteger)maxLongEdge {
    @autoreleasepool {
        cv::Mat gray = uiimageToMatGray(image);
        if (gray.empty()) { return 0.0; }

        cv::Mat resized;
        resizeKeepingLongEdge(gray, resized, (int)maxLongEdge);

        cv::Mat m = uiimageMaskToMatBinary(mask, resized.size());
        if (m.empty()) { return 0.0; }

        int k = (int)ksize;
        if (k != 1 && k != 3 && k != 5) { k = 3; }

        cv::Mat gx, gy;
        cv::Sobel(resized, gx, CV_32F, 1, 0, k);
        cv::Sobel(resized, gy, CV_32F, 0, 1, k);

        cv::Mat mag2;
        cv::multiply(gx, gx, gx);
        cv::multiply(gy, gy, gy);
        cv::add(gx, gy, mag2);

        return maskedMean(mag2, m);
    }
}

+ (double)laplacianVarianceFor:(UIImage *)image mask:(UIImage *)mask {
    @autoreleasepool {
        cv::Mat gray = uiimageToMatGray(image);
        if (gray.empty()) { return 0.0; }

        cv::Mat m = uiimageMaskToMatBinary(mask, gray.size());
        if (m.empty()) { return 0.0; }

        cv::Mat lap;
        cv::Laplacian(gray, lap, CV_32F, 3);
        return maskedVariance(lap, m);
    }
}

@end
