import AVFoundation
import CoreML
import UIKit

final class CameraViewModel: NSObject, ObservableObject {
    // Camera
    let cameraService = CameraService()

    // UI state
    @Published var isProcessing = false
    @Published var capturedImage: UIImage?
    @Published var showingPreview = false

    // Help overlay
    @Published var failedAttempts = 0
    @Published var showHelpOverlay = false

    // Banner
    @Published var showBanner = false
    @Published var bannerMessage: String?
    @Published var bannerIsSuccess = false

    // Segmentation + overlay
    private let segmenter = LungSegmentationService()
    @Published var overlayMask: UIImage?
    @Published var isCheckingSharpness = false

    // Sharpness thresholds (tune on masked scores)
    private let sobelKernelSize: Int32 = 3
    private let sobelThresh: Double = 3500.0
    private let lapThresh:   Double = 750.0

    // Segmentation params
    private let segThreshold: Float = 0.05
    private let segDilation: Int = 12

    // (Optional) coverage sanity window to avoid bogus “all white” masks
    private let minCoverage: Double = 0.05     // 5%
    private let maxCoverage: Double = 0.75     // 75%

    private lazy var segModel: LungSegmentation? = {
        return try? LungSegmentation(configuration: MLModelConfiguration())
    }()

    override init() {
        super.init()
        cameraService.delegate = self
        cameraService.startSession()
    }

    deinit {
        cameraService.stopSession()
    }

    // MARK: - Camera controls

    func setZoom(factor: CGFloat)  { cameraService.setZoom(factor: factor) }
    func setExposure(value: Float) { cameraService.setExposure(value: value) }

    func capturePhoto() {
        guard !isProcessing else { return }
        isProcessing = true
        cameraService.capturePhoto()
    }

    // MARK: - Banner

    private func showBannerMessage(success: Bool, message: String) {
        bannerIsSuccess = success
        bannerMessage = message
        UINotificationFeedbackGenerator().notificationOccurred(success ? .success : .warning)
        showBanner = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
            self?.showBanner = false
        }
    }

    // MARK: - Strict gate: segment → masked sharpness → pass/fail

    private func evaluateMaskedSharpnessAndProceed(_ image: UIImage) {
        isCheckingSharpness = true
        overlayMask = nil

        guard segModel != nil else {
            DispatchQueue.main.async {
                self.isCheckingSharpness = false
                self.showBannerMessage(success: false, message: "Segmentation model missing.")
                self.isProcessing = false
                self.recordCaptureResult(passed: false)
            }
            return
        }

        // Heavy work off-main
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // Run the new segmenter
                let report = try self.segmenter.analyze(
                    image: image,
                    threshold: self.segThreshold,
                    dilationRadius: self.segDilation
                ) { chwInput in
                    // Only accept grayscale image input (OneComponent8)
                    guard let core = self.segModel?.model else {
                        throw NSError(domain: "Seg", code: -1,
                                      userInfo: [NSLocalizedDescriptionKey: "Segmentation model not loaded"])
                    }

                    // Expect exactly one IMAGE input
                    guard let (inName, inDesc) = core.modelDescription.inputDescriptionsByName.first,
                          inDesc.type == .image,
                          let constraint = inDesc.imageConstraint,
                          constraint.pixelFormatType == kCVPixelFormatType_OneComponent8
                    else {
                        throw NSError(domain: "Seg", code: -2,
                                      userInfo: [NSLocalizedDescriptionKey: "Model input must be grayscale (OneComponent8) image"])
                    }

                    // Use model-declared size if available, otherwise 512
                    let width  = Int(constraint.pixelsWide > 0 ? constraint.pixelsWide : 512)
                    let height = Int(constraint.pixelsHigh > 0 ? constraint.pixelsHigh : 512)

                    let resized = image.resized(to: CGSize(width: width, height: height))
                    guard let pb = resized.pixelBufferGray(width: width, height: height) else {
                        throw NSError(domain: "Seg", code: -3,
                                      userInfo: [NSLocalizedDescriptionKey: "Failed to make OneComponent8 pixel buffer"])
                    }

                    let provider = try MLDictionaryFeatureProvider(dictionary: [
                        inName: MLFeatureValue(pixelBuffer: pb)
                    ])

                    // Predict
                    let result = try core.prediction(from: provider)

                    // Return the first MultiArray output
                    if let outName = core.modelDescription.outputDescriptionsByName
                        .first(where: { $0.value.type == .multiArray })?.key,
                       let arr = result.featureValue(for: outName)?.multiArrayValue {
                        return arr
                    }

                    throw NSError(domain: "Seg", code: -4,
                                  userInfo: [NSLocalizedDescriptionKey: "No MultiArray output found"])
                }

                // Quick coverage sanity (pre-dilate, at 512)
                let coverage = report.coveragePreDilate
                print("[CameraVM] lung coverage (pre-dilate,512): \(Int(coverage * 100))%")

                if coverage < self.minCoverage || coverage > self.maxCoverage {
                    DispatchQueue.main.async {
                        self.isCheckingSharpness = false
                        self.overlayMask = report.maskOriginalSize
                        self.showBannerMessage(success: false, message: "Mask unreliable. Retake.")
                        self.isProcessing = false
                        self.recordCaptureResult(passed: false)
                    }
                    return
                }

                // Edge metrics already computed by segmenter (on original image, masked)
                let ten = report.tenengradInsideMask
                let lap = report.laplacianVarInsideMask
                let pass = (ten >= self.sobelThresh) || (lap >= self.lapThresh)

                DispatchQueue.main.async {
                    self.overlayMask = report.maskOriginalSize
                    self.isCheckingSharpness = false
                    if !pass {
                        self.showBannerMessage(success: false, message: "Region blurry. Please retake.")
                        self.isProcessing = false
                        self.recordCaptureResult(passed: false)
                        return
                    }
                    self.capturedImage = image
                    self.showingPreview = true
                    self.isProcessing = false
                    self.recordCaptureResult(passed: true)
                }
            } catch {
                print("[CameraVM] segmentation error:", error.localizedDescription)
                DispatchQueue.main.async {
                    self.isCheckingSharpness = false
                    self.showBannerMessage(success: false, message: "Segmentation failed. Retake.")
                    self.isProcessing = false
                    self.recordCaptureResult(passed: false)
                }
            }
        }
    }


    // MARK: - Attempts / Help overlay

    func recordCaptureResult(passed: Bool) {
        if passed {
            failedAttempts = 0
            showHelpOverlay = false
        } else {
            failedAttempts += 1
            if failedAttempts >= 5 { showHelpOverlay = true }
        }
    }

    func dismissHelpOverlay() {
        showHelpOverlay = false
        failedAttempts = 0
    }
}

// MARK: - AVCapturePhotoCaptureDelegate

extension CameraViewModel: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }

            if let error = error {
                print("Error capturing photo:", error)
                self.showBannerMessage(success: false, message: "Capture failed. Try again.")
                self.isProcessing = false
                return
            }

            guard let data = photo.fileDataRepresentation(),
                  let image = UIImage(data: data) else {
                self.showBannerMessage(success: false, message: "Could not read photo data.")
                self.isProcessing = false
                return
            }

            // 🔒 STRICT MASKED GATE
            self.evaluateMaskedSharpnessAndProceed(image)
        }
    }
}
