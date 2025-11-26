//
//  CameraView.swift
//  ChestX-RayImaging
//
//  Created by Connor McMahon on 3/20/25.
//

import AVFoundation
import UIKit

class CameraService: NSObject {
    private var session: AVCaptureSession?
    private var device: AVCaptureDevice?
    private let output = AVCapturePhotoOutput()
    let previewLayer = AVCaptureVideoPreviewLayer()
    weak var delegate: AVCapturePhotoCaptureDelegate?
    
    // Track current camera settings
    private(set) var currentZoomFactor: CGFloat = 1.0
    private(set) var currentISO: Float = 0
    private(set) var currentExposureValue: Float = 0  // Store the slider value
    private(set) var currentExposureDuration: Double = 0
    
    func startSession() {
        checkPermissions { error in
            if let error = error {
                print("Error starting camera: \(error)")
                return
            }
        }
    }
    
    private func checkPermissions(completion: @escaping (Error?) -> ()) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setupCamera(completion: completion)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard granted else {
                    completion(NSError(domain: "CameraService", code: 1, userInfo: [NSLocalizedDescriptionKey: "Camera access denied"]))
                    return
                }
                self?.setupCamera(completion: completion)
            }
        case .denied, .restricted:
            completion(NSError(domain: "CameraService", code: 1, userInfo: [NSLocalizedDescriptionKey: "Camera access denied"]))
        @unknown default:
            completion(NSError(domain: "CameraService", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unknown camera authorization status"]))
        }
    }
    
    private func setupCamera(completion: @escaping (Error?) -> ()) {
        let session = AVCaptureSession()
        if let device = AVCaptureDevice.default(for: .video) {
            self.device = device
            do {
                session.sessionPreset = .photo
                
                let input = try AVCaptureDeviceInput(device: device)
                if session.canAddInput(input) {
                    session.addInput(input)
                }
                
                if session.canAddOutput(output) {
                    session.addOutput(output)
                    output.maxPhotoQualityPrioritization = .quality
                }
                
                // Configure preview layer
                previewLayer.session = session
                previewLayer.videoGravity = .resizeAspect
                
                // Start the session on a background thread
                DispatchQueue.global(qos: .userInitiated).async {
                    session.startRunning()
                    DispatchQueue.main.async {
                        self.session = session
                        // Set initial camera settings with lower exposure
                        self.resetCameraSettings()
                        completion(nil)
                    }
                }
                
            } catch {
                completion(error)
            }
        }
    }
    
    func resetCameraSettings() {
        guard let device = device else { return }
        do {
            try device.lockForConfiguration()
            // Reset zoom to 1.0 (no zoom)
            device.videoZoomFactor = 1.0
            currentZoomFactor = 1.0
            
            // Set exposure to a lower value
            setExposure(value: -0.5)
            currentExposureValue = -0.5
            
            device.unlockForConfiguration()
        } catch {
            print("Error resetting camera settings: \(error)")
        }
    }
    
    func setZoom(factor: CGFloat) {
        guard let device = device else { return }
        do {
            try device.lockForConfiguration()
            device.videoZoomFactor = factor
            currentZoomFactor = factor
            device.unlockForConfiguration()
        } catch {
            print("Error setting zoom: \(error)")
        }
    }
    
    func setExposure(value: Float) {
        guard let device = device else { return }
        do {
            try device.lockForConfiguration()
            
            // Store the slider value
            currentExposureValue = value
            
            // Calculate new ISO based on the slider value
            let isoRange = device.activeFormat.minISO...device.activeFormat.maxISO
            let newISO = isoRange.lowerBound + (isoRange.upperBound - isoRange.lowerBound) * (value + 1) / 2
            device.setExposureModeCustom(duration: device.exposureDuration, iso: newISO) { _ in }
            
            currentISO = newISO
            currentExposureDuration = device.exposureDuration.seconds
            
            device.unlockForConfiguration()
        } catch {
            print("Error setting exposure: \(error)")
        }
    }
    
    func capturePhoto() {
        guard let session = session, session.isRunning, let delegate = delegate else { return }
        
        let settings = AVCapturePhotoSettings()
        output.capturePhoto(with: settings, delegate: delegate)
    }
    
    func stopSession() {
        session?.stopRunning()
    }
}
