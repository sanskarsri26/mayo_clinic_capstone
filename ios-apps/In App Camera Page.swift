//
//  CameraView.swift
//  ChestX-RayImaging
//
//  Created by Connor McMahon on 3/20/25.
//

import SwiftUI
import AVFoundation

struct CameraView: View {
    let dismiss: () -> Void
    @StateObject private var viewModel = CameraViewModel()
    @State private var zoom: CGFloat = 1.0
    @State private var exposure: Float = 0.0
    @State private var showingPreview = false
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Live Camera Preview
                CameraPreview(cameraService: viewModel.cameraService)
                    .ignoresSafeArea()

                // Camera Controls + Top Bar
                VStack {
                    // Top Bar
                    HStack {
                        Button(action: { dismiss() }) {
                            Image(systemName: "xmark")
                                .font(.title2)
                                .foregroundColor(.white)
                                .padding()
                        }
                        Spacer()

                        // Small inline status when analyzing
                        if viewModel.isProcessing || viewModel.isCheckingSharpness {
                            HStack(spacing: 8) {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                Text(viewModel.isCheckingSharpness ? "Analyzing…" : "Processing…")
                                    .foregroundColor(.white)
                                    .font(.subheadline)
                            }
                            .padding(.trailing, 12)
                        }
                    }
                    .background(Color.black.opacity(0.5))

                    Spacer()

                    // Camera Controls
                    VStack(spacing: 20) {
                        // Zoom Slider
                        HStack {
                            Image(systemName: "magnifyingglass")
                                .foregroundColor(.white)
                            Slider(value: $zoom, in: 1.0...5.0) { _ in
                                viewModel.setZoom(factor: zoom)
                            }
                            .accentColor(.white)
                        }

                        // Exposure Slider
                        HStack {
                            Image(systemName: "camera.aperture")
                                .foregroundColor(.white)
                            Slider(value: $exposure, in: -1.0...1.0) { _ in
                                viewModel.setExposure(value: exposure)
                            }
                            .accentColor(.white)
                        }

                        // Capture Button (disabled while processing or checking sharpness)
                        Button(action: { viewModel.capturePhoto() }) {
                            Circle()
                                .fill((viewModel.isProcessing || viewModel.isCheckingSharpness) ? Color.gray : Color.white)
                                .frame(width: 65, height: 65)
                                .overlay(
                                    Circle()
                                        .stroke(Color.white.opacity(0.9), lineWidth: 2)
                                        .frame(width: 75, height: 75)
                                )
                                .overlay(
                                    // Optional icon change while busy
                                    Group {
                                        if viewModel.isProcessing || viewModel.isCheckingSharpness {
                                            ProgressView()
                                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                        }
                                    }
                                )
                        }
                        .disabled(viewModel.isProcessing || viewModel.isCheckingSharpness)
                        .padding(.top, 20)
                    }
                    .padding()
                    .background(Color.black.opacity(0.5))
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                // Alignment Guide
                VStack {
                    Spacer()
                    ZStack {
                        // Alignment Text
                        Text("Align the X-Ray image inside the lines")
                            .foregroundColor(.white)
                            .font(.system(size: 16, weight: .medium))
                            .padding(.horizontal, 20)
                            .padding(.vertical, 8)
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(8)
                            .position(x: geometry.size.width / 2, y: geometry.size.height * 0.1)

                        // Bounding Box
                        Rectangle()
                            .stroke(Color.white, lineWidth: 2)
                            .frame(width: geometry.size.width * 0.6,
                                   height: geometry.size.width * 0.8)
                            .position(x: geometry.size.width / 2, y: geometry.size.height * 0.5)
                    }
                    Spacer()
                }
                .allowsHitTesting(false)

                // Help Overlay (Auto-shows after repeated failures)
                if viewModel.showHelpOverlay {
                    HelpOverlay { viewModel.dismissHelpOverlay() }
                        .transition(.opacity.combined(with: .scale))
                        .zIndex(10)
                }

                
                // Banner (reused for sharpness failure / other notices)
                if viewModel.showBanner, let msg = viewModel.bannerMessage {
                    VStack {
                        BannerView(message: msg, isSuccess: viewModel.bannerIsSuccess)
                            .padding(.top, 12)
                        Spacer()
                    }
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .animation(.easeInOut(duration: 0.2), value: viewModel.showBanner)
                }
            }
        }
        .navigationBarBackButtonHidden(true)
        .navigationDestination(isPresented: $viewModel.showingPreview) {
            if let image = viewModel.capturedImage {
                ImagePreviewView(image: image, dismiss: dismiss)
            }
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    let cameraService: CameraService
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        cameraService.previewLayer.frame = view.bounds
        view.layer.addSublayer(cameraService.previewLayer)
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            self.cameraService.previewLayer.frame = uiView.bounds
        }
    }
}

struct BannerView: View {
    let message: String
    let isSuccess: Bool

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: isSuccess ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                .foregroundColor(.white)
            Text(message)
                .foregroundColor(.white)
                .font(.subheadline)
                .lineLimit(2)
                .multilineTextAlignment(.leading)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(isSuccess ? Color.green.opacity(0.95) : Color.red.opacity(0.95))
        .cornerRadius(12)
        .shadow(radius: 4, y: 2)
        .padding(.horizontal, 16)
    }
}


