import Foundation
import CoreML
import Vision
import UIKit

class ModelService {
    private var model: MLModel?
    private var classifierWeights: [Float]?
    
    // CheXpert labels
    private let conditions = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices"
    ]
    
    init() {
        loadModel()
        loadClassifierWeights()
    }
    
    private func loadModel() {
        if let modelURL = Bundle.main.url(forResource: "CheXpert_DenseNet121_FP32_multioutput", withExtension: "mlmodelc") {
            do {
                model = try MLModel(contentsOf: modelURL)
            } catch {
                print("Error loading model: \(error)")
            }
        } else {
            print("Failed to find compiled model in bundle")
        }
    }
    
    private func loadClassifierWeights() {
        if let weightsURL = Bundle.main.url(forResource: "classifier_weights", withExtension: "npy") {
            do {
                let data = try Data(contentsOf: weightsURL)
                classifierWeights = data.withUnsafeBytes { buffer in
                    Array(buffer.bindMemory(to: Float.self))
                }
            } catch {
                print("Error loading weights: \(error)")
            }
        } else {
            print("Failed to find weights in bundle")
        }
    }
    
    func analyzeImage(_ image: UIImage, completion: @escaping (Float, UIImage?, [String: Float]) -> Void) {
        guard let model = model, let classifierWeights = classifierWeights else {
            print("Model or weights not loaded!")
            DispatchQueue.main.async {
                completion(0.0, nil, [:])
            }
            return
        }
        
        // Ensure image is in RGB format and properly sized
        let resizedImage = image.resized(to: CGSize(width: 224, height: 224))
        
        guard let pixelBuffer = resizedImage.pixelBuffer(width: 224, height: 224) else {
            print("Failed to convert image to pixel buffer")
            DispatchQueue.main.async {
                completion(0.0, nil, [:])
            }
            return
        }
        
        // Create a strong reference to self for the async block
        let strongSelf = self
        
        DispatchQueue.global(qos: .userInitiated).async(execute: DispatchWorkItem(block: {
            do {
                // Prepare input dictionary
                let input = try MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(pixelBuffer: pixelBuffer)])
                
                let result = try model.prediction(from: input)
                
                guard let featureMap = result.featureValue(for: "features_1")?.multiArrayValue,
                      let predictions = result.featureValue(for: "var_1915")?.multiArrayValue else {
                    print("Failed to get outputs from prediction")
                    DispatchQueue.main.async {
                        completion(0.0, nil, [:])
                    }
                    return
                }
                
                // Safely convert predictions to array
                var predictionArray = [Float](repeating: 0, count: 14)
                for i in 0..<min(14, predictions.count) {
                    predictionArray[i] = predictions[i].floatValue
                }
                
                // Calculate CAM with safety checks
                let (cam, _) = strongSelf.computeCAM(
                    predictions: predictionArray,
                    features: featureMap,
                    classifierWeights: classifierWeights
                )
                
                // Create visualization
                let visualization = strongSelf.createGradCAMVisualization(
                    image: image,
                    cam: cam,
                    alpha: 0.5
                )
                
                // Process predictions safely
                var predictionResults: [String: Float] = [:]
                var maxProbability: Float = 0.0
                
                for (index, condition) in strongSelf.conditions.enumerated() {
                    if index < predictionArray.count {
                        let probability = predictionArray[index]
                        predictionResults[condition] = probability
                        maxProbability = max(maxProbability, probability)
                    }
                }
                
                DispatchQueue.main.async {
                    completion(maxProbability, visualization, predictionResults)
                }
                
            } catch {
                print("Error during prediction: \(error)")
                DispatchQueue.main.async {
                    completion(0.0, nil, [:])
                }
            }
        }))
    }
    
    private func computeCAM(predictions: [Float], features: MLMultiArray, classifierWeights: [Float]) -> ([Float], Int) {
        // Get dimensions
        let channels = features.shape[1].intValue  // C=1024
        let height = features.shape[2].intValue    // H=7
        let width = features.shape[3].intValue     // W=7
        
        // Safely get the predicted class
        let predClass = predictions.indices.max(by: { predictions[$0] < predictions[$1] }) ?? 0
        
        // Safely calculate weight vector indices
        let startIdx = predClass * channels
        let endIdx = min((predClass + 1) * channels, classifierWeights.count)
        guard startIdx < classifierWeights.count, endIdx <= classifierWeights.count else {
            // Return empty result if indices are invalid
            return ([Float](repeating: 0, count: height * width), predClass)
        }
        
        // Get weights for predicted class
        let weightVector = Array(classifierWeights[startIdx..<endIdx])
        
        // Create flattened result array
        var flatCam = [Float](repeating: 0, count: height * width)
        
        // Safely compute weighted sum
        for h in 0..<height {
            for w in 0..<width {
                var sum: Float = 0
                for c in 0..<channels {
                    // Safely access feature value
                    let indices = [0, c, h, w] as [NSNumber]
                    guard indices.count == features.shape.count else { continue }
                    
                    let value = features[indices].floatValue
                    // Apply ReLU
                    let feature = max(value, 0)
                    
                    // Safely access weight
                    guard c < weightVector.count else { continue }
                    sum += feature * weightVector[c]
                }
                // Store in flattened array
                let idx = h * width + w
                guard idx < flatCam.count else { continue }
                flatCam[idx] = sum
            }
        }
        
        // Apply ReLU to result
        flatCam = flatCam.map { max($0, 0) }
        
        // Normalize to [0, 1]
        if let minVal = flatCam.min(), let maxVal = flatCam.max(), maxVal > minVal {
            flatCam = flatCam.map { ($0 - minVal) / (maxVal - minVal) }
        }
        
        return (flatCam, predClass)
    }
    
    private func createGradCAMVisualization(image: UIImage, cam: [Float], alpha: Float = 0.5) -> UIImage? {
        // Ensure we're working with an upright image
        let uprightImage = image.fixOrientation()
        let originalSize = uprightImage.size
        let width = Int(originalSize.width)
        let height = Int(originalSize.height)
        
        // Create contexts for original image and heatmap
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(data: nil,
                                    width: width,
                                    height: height,
                                    bitsPerComponent: 8,
                                    bytesPerRow: width * 4,
                                    space: colorSpace,
                                    bitmapInfo: bitmapInfo.rawValue) else {
            return nil
        }
        
        // Draw original image
        context.draw(uprightImage.cgImage!, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Create heatmap context
        guard let heatmapContext = CGContext(data: nil,
                                           width: width,
                                           height: height,
                                           bitsPerComponent: 8,
                                           bytesPerRow: width * 4,
                                           space: colorSpace,
                                           bitmapInfo: bitmapInfo.rawValue) else {
            return nil
        }
        
        // Resize CAM to image dimensions
        let heatmapSize = Int(sqrt(Float(cam.count)))
        let scaleX = Float(width) / Float(heatmapSize)
        let scaleY = Float(height) / Float(heatmapSize)
        
        // Apply JET colormap and draw heatmap
        for y in 0..<height {
            for x in 0..<width {
                // Get interpolated value
                let heatmapX = Float(x) / scaleX
                let heatmapY = Float(y) / scaleY
                
                let x0 = Int(floor(heatmapX))
                let x1 = min(x0 + 1, heatmapSize - 1)
                let y0 = Int(floor(heatmapY))
                let y1 = min(y0 + 1, heatmapSize - 1)
                
                let wx = heatmapX - Float(x0)
                let wy = heatmapY - Float(y0)
                
                let v00 = cam[y0 * heatmapSize + x0]
                let v10 = cam[y0 * heatmapSize + x1]
                let v01 = cam[y1 * heatmapSize + x0]
                let v11 = cam[y1 * heatmapSize + x1]
                
                let value = (1 - wx) * (1 - wy) * v00 +
                           wx * (1 - wy) * v10 +
                           (1 - wx) * wy * v01 +
                           wx * wy * v11
                
                // Apply JET colormap
                var r: CGFloat = 0
                var g: CGFloat = 0
                var b: CGFloat = 0
                
                let cgValue = CGFloat(value)
                
                if cgValue < 0.125 {
                    b = 0.5 + 4 * cgValue
                } else if cgValue < 0.375 {
                    b = 1.0
                    g = (cgValue - 0.125) * 4
                } else if cgValue < 0.625 {
                    b = 1.0 + 4 * (0.375 - cgValue)
                    g = 1.0
                    r = 4 * (cgValue - 0.375)
                } else if cgValue < 0.875 {
                    g = 1.0 + 4 * (0.625 - cgValue)
                    r = 1.0
                } else {
                    r = 1.0 + 4 * (0.875 - cgValue)
                }
                
                let color = UIColor(red: r, green: g, blue: b, alpha: CGFloat(alpha))
                heatmapContext.setFillColor(color.cgColor)
                heatmapContext.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
        }
        
        // Blend original image with heatmap
        context.setBlendMode(.normal)
        if let heatmapImage = heatmapContext.makeImage() {
            context.draw(heatmapImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        
        // Create final image
        guard let finalImage = context.makeImage() else {
            return nil
        }
        
        return UIImage(cgImage: finalImage)
    }
}

extension UIImage {
    func resized(to size: CGSize) -> UIImage {
        return UIGraphicsImageRenderer(size: size).image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
    }
    
    func fixOrientation() -> UIImage {
        if imageOrientation == .up {
            return self
        }
        
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        draw(in: CGRect(origin: .zero, size: size))
        let normalizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return normalizedImage ?? self
    }
    
    func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let uprightImage = fixOrientation()
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                       width,
                                       height,
                                       kCVPixelFormatType_32ARGB,
                                       nil,
                                       &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                              width: width,
                              height: height,
                              bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        guard let cgContext = context else {
            return nil
        }
        
        cgContext.draw(uprightImage.cgImage!, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return buffer
    }
    
    func pixelBufferGray(width: Int, height: Int) -> CVPixelBuffer? {
            let upright = fixOrientation()

            // Allocate grayscale pixel buffer
            var pixelBuffer: CVPixelBuffer?
            let attrs: [CFString: Any] = [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                kCVPixelBufferIOSurfacePropertiesKey: [:]
            ]
            let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                             width, height,
                                             kCVPixelFormatType_OneComponent8,
                                             attrs as CFDictionary,
                                             &pixelBuffer)
            guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

            CVPixelBufferLockBaseAddress(buffer, [])
            defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

            guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
            let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

            // Draw the image into a DeviceGray context that writes directly into the buffer
            let cs = CGColorSpaceCreateDeviceGray()
            guard let ctx = CGContext(data: base,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: cs,
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue)
            else { return nil }

            ctx.interpolationQuality = .high

            if let cg = upright.cgImage {
                ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
            } else {
                // Fallback (rare): rasterize via UIKit then draw
                UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), true, 1)
                upright.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
                let tmp = UIGraphicsGetImageFromCurrentImageContext()
                UIGraphicsEndImageContext()
                if let cg2 = tmp?.cgImage {
                    ctx.draw(cg2, in: CGRect(x: 0, y: 0, width: width, height: height))
                } else {
                    return nil
                }
            }

            return buffer
        }
}
