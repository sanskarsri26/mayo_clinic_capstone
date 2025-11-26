import Foundation
import UIKit
import CoreML
import Accelerate

/// Minimal lung segmentation runner:
/// 1) Preprocess to 3x512x512 (ImageNet normalization)
/// 2) Inference via injected closure (returns [1,2,512,512] or [1,1,512,512])
/// 3) Softmax/threshold to binary lung mask (0/255) at 512
/// 4) Coverage on **pre-dilate, pre-resize** mask
/// 5) Optional dilation for edge work only
/// 6) Resize mask to original image with **nearest neighbor**
/// 7) Tenengrad (Sobel) & Laplacian variance inside mask (OpenCVWrapper)
final class LungSegmentationService {

    struct Report {
        let mask512: UIImage                // 0/255 at 512x512 (binary)
        let maskOriginalSize: UIImage       // 0/255 resized (nearest) to input.size
        let coveragePreDilate: Double       // 0..1 measured on mask512 BEFORE dilation
        let tenengradInsideMask: Double
        let laplacianVarInsideMask: Double
    }

    /// Type of your inference callback.
    /// Input: MLMultiArray Float32 [1,3,512,512]
    /// Output: MLMultiArray Float32 either [1,2,512,512] (logits or probs) or [1,1,512,512] (prob)
    typealias Infer = (_ chwInput: MLMultiArray) throws -> MLMultiArray

    /// Main entry.
    /// - threshold: probability cutoff (default 0.2)
    /// - dilationRadius: pixel radius of binary dilation (ONLY used for edge metrics; coverage ignores it)
    func analyze(
        image: UIImage,
        threshold: Float,
        dilationRadius: Int,
        infer: Infer
    ) throws -> Report {

        // 1) Preprocess to CHW Float32 [1,3,512,512] with ImageNet normalization
        let (chw, _) = try Self.makeCHWInput(from: image)

        // 2) Run model
        let rawOut = try infer(chw)

        // 3) Extract lung probability map at 512x512 (Float32)
        let lungProb = try Self.extractLungProbability(from: rawOut) // [512*512] Float32 row-major

        // 4) Threshold → strictly binary 0/255 mask
        let bin512 = Self.thresholdToMask(prob: lungProb, width: 512, height: 512, cutoff: threshold)

        // 5) Coverage on pre-dilate, pre-resize mask
        let coverage = Self.binaryCoverage(mask: bin512)

        // 6) Resize binary mask to original image size using nearest neighbor
        let maskOriginal = Self.resizeBinaryNearest(bin512, to: image.size) ?? bin512

        // 7) Edge metrics inside mask (original image + original-size mask)
        //    Note: OpenCVWrapper expects 0/255 grayscale mask (which we supply).
        let tng = OpenCVWrapper.tenengrad(for: image, mask: maskOriginal, kernel: 3, maxLongEdge: 0)
        let lap = OpenCVWrapper.laplacianVariance(for: image, mask: maskOriginal)

        // Logs for quick triage
        print("[LungSeg] size(in)=\(Int(image.size.width))x\(Int(image.size.height))  size(512)=512x512")
        print("[LungSeg] coverage(pre-dilate,512) = \(Int(coverage*100))%  threshold=\(threshold)")
        print("[LungSeg] dilation=\(dilationRadius)  tenengrad=\(String(tng))  lapVar=\(String(lap))")

        return Report(mask512: bin512,
                      maskOriginalSize: maskOriginal,
                      coveragePreDilate: coverage,
                      tenengradInsideMask: tng,
                      laplacianVarInsideMask: lap)
    }

    // MARK: - Preprocess (ImageNet normalization, CHW)

    /// Returns (MLMultiArray [1,3,512,512], the 512x512 RGB UIImage actually fed)
    private static func makeCHWInput(from image: UIImage) throws -> (MLMultiArray, UIImage) {
        // 1) Resize to 512x512 (bilinear OK for the *input* image)
        guard let rgb512 = resizeToRGB(image, size: CGSize(width: 512, height: 512)) else {
            throw NSError(domain: "LungSeg", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to resize input to 512"])
        }

        // 2) Read bytes
        guard let cg = rgb512.cgImage else {
            throw NSError(domain: "LungSeg", code: -2, userInfo: [NSLocalizedDescriptionKey: "No CGImage"])
        }
        let w = cg.width, h = cg.height
        var data = [UInt8](repeating: 0, count: w * h * 4) // RGBA8
        guard let ctx = CGContext(data: &data,
                                  width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            throw NSError(domain: "LungSeg", code: -3, userInfo: [NSLocalizedDescriptionKey: "CGContext fail"])
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

        // 3) Allocate CHW [1,3,512,512]
        let shape: [NSNumber] = [1, 3, 512, 512]
        let arr = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)

        // ImageNet normalization
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std:  [Float] = [0.229, 0.224, 0.225]

        // Fill as CHW
        // Index mapping: [b,c,y,x] with b=0
        func idx(_ c: Int, _ y: Int, _ x: Int) -> Int {
            return c * (h * w) + y * w + x
        }

        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(arr.dataPointer))
        for y in 0..<h {
            for x in 0..<w {
                let base = (y * w + x) * 4
                // If original was grayscale, r==g==b already; if RGB, use channels
                let r = Float(data[base + 0]) / 255.0
                let g = Float(data[base + 1]) / 255.0
                let b = Float(data[base + 2]) / 255.0
                ptr[idx(0, y, x)] = (r - mean[0]) / std[0]
                ptr[idx(1, y, x)] = (g - mean[1]) / std[1]
                ptr[idx(2, y, x)] = (b - mean[2]) / std[2]
            }
        }
        return (arr, rgb512)
    }

    /// Resize to 512x512 and force RGB colorspace (grayscale is replicated to RGB).
    private static func resizeToRGB(_ image: UIImage, size: CGSize) -> UIImage? {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        return renderer.image { ctx in
            ctx.cgContext.interpolationQuality = .high
            image.draw(in: CGRect(origin: .zero, size: size))
        }
    }

    // MARK: - Output handling

    /// Accepts [1,2,512,512] (logits or probs) OR [1,1,512,512] (prob).
    /// Returns flattened [512*512] probabilities for lung.
    private static func extractLungProbability(from out: MLMultiArray) throws -> [Float] {
        let n = out.count                // expect 512*512
        precondition(out.dataType == .float16 && n == 512*512)
        var vals = [Float](repeating: 0, count: n)
        vals.withUnsafeMutableBufferPointer { dst in
            var src = vImage_Buffer(data: out.dataPointer, height: 1, width: UInt(n), rowBytes: n * 2)
            var dstB = vImage_Buffer(data: dst.baseAddress, height: 1, width: UInt(n), rowBytes: n * 4)
            _ = vImageConvert_Planar16FtoPlanarF(&src, &dstB, 0)
        }
        // clamp
        for i in 0..<vals.count { vals[i] = min(1, max(0, vals[i])) }
        
        // Optional debug stats
        if let mn = vals.min(), let mx = vals.max() {
            let mean = vals.reduce(0,+) / Float(vals.count)
            print("[LungSeg] prob stats: min=\(mn) max=\(mx) mean=\(mean)")
        }
        
        return vals
    }

    private static func thresholdToMask(prob: [Float], width: Int, height: Int, cutoff: Float) -> UIImage {
        var buf = [UInt8](repeating: 0, count: width * height)
        for i in 0..<width*height {
            buf[i] = (prob[i] >= cutoff) ? 255 : 0
        }
        return grayImage(from: buf, width: width, height: height)
    }

    private static func binaryCoverage(mask: UIImage) -> Double {
        guard let cg = mask.cgImage else { return 0 }
        let w = cg.width, h = cg.height
        var buf = [UInt8](repeating: 0, count: w * h)
        guard let ctx = CGContext(data: &buf, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return 0 }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        var white = 0
        for px in buf { if px >= 128 { white += 1 } }
        return Double(white) / Double(w * h)
    }

    // MARK: - Dilation & Resize (binary-safe)

    private static func vimageDilate(mask: UIImage, radius: Int) -> UIImage? {
        guard let cg = mask.cgImage else { return nil }
        let w = cg.width, h = cg.height
        var src = [UInt8](repeating: 0, count: w * h)
        var dst = [UInt8](repeating: 0, count: w * h)

        guard let ctx = CGContext(data: &src, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        ctx.interpolationQuality = .none
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

        var s = vImage_Buffer(data: &src, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
        var d = vImage_Buffer(data: &dst, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)

        let dia = 2 * radius + 1
        let kernel = [UInt8](repeating: 1, count: dia * dia)

        vImageDilate_Planar8(&s, &d, 0, 0, kernel, vImagePixelCount(UInt32(dia)), vImagePixelCount(UInt32(dia)), vImage_Flags(kvImageEdgeExtend))
        return grayImage(from: dst, width: w, height: h)
    }

    /// Nearest-neighbor resize for **binary** masks (keeps 0/255 only)
    private static func resizeBinaryNearest(_ mask: UIImage, to size: CGSize) -> UIImage? {
        guard let cg = mask.cgImage else { return nil }
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        return renderer.image { ctx in
            ctx.cgContext.interpolationQuality = .none
            UIImage(cgImage: cg).draw(in: CGRect(origin: .zero, size: size))
        }
    }

    private static func grayImage(from buf: [UInt8], width: Int, height: Int) -> UIImage {
        var data = buf
        let cs = CGColorSpaceCreateDeviceGray()
        let provider = CGDataProvider(data: NSData(bytes: &data, length: data.count))!
        let cg = CGImage(width: width, height: height,
                         bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
                         space: cs, bitmapInfo: CGBitmapInfo(rawValue: 0),
                         provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        return UIImage(cgImage: cg, scale: 1, orientation: .up)
    }
}

// MARK: - MLMultiArray helper

private extension MLMultiArray {
    func toFloatArray() -> [Float] {
        let n = self.count
        switch self.dataType {
        case .float32:
            let p = self.dataPointer.bindMemory(to: Float32.self, capacity: n)
            return Array(UnsafeBufferPointer(start: p, count: n))
            
        case .double:
            let p = self.dataPointer.bindMemory(to: Double.self, capacity: n)
            var out = [Float](repeating: 0, count: n)
            for i in 0..<n { out[i] = Float(p[i]) }
            return out
            
        case .float16:
            var out = [Float](repeating: 0, count: n)
            var src = vImage_Buffer(data: self.dataPointer, height: 1, width: UInt(n), rowBytes: n * 2)
            var dst = vImage_Buffer(data: &out,            height: 1, width: UInt(n), rowBytes: n * 4)
            let err = vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
            if err == kvImageNoError { return out }
            return out // fallback (zeros) if conversion fails
            
        case .int32:
            let p = self.dataPointer.bindMemory(to: Int32.self, capacity: n)
            return (0..<n).map { Float(p[$0]) }
            
        default:
            return []
        }
    }
}
