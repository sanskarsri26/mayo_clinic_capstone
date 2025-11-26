//
//  ContentView.swift
//  ChestX-RayImaging
//
//  Created by Connor McMahon on 3/20/25.
//


import SwiftUI
import CoreData
import AVFoundation

struct ContentView: View {
    @State private var showingCamera = false
    @State private var showingPermissionAlert = false
    @State private var capturedImage: UIImage?
    @State private var showingPreview = false
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                Image(systemName: "lungs.fill")
                    .font(.system(size: 60))
                    .foregroundColor(.blue)
                
                Text("Chest X-Ray Imaging")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Text("Capture and analyze chest X-ray images")
                    .font(.subheadline)
                    .foregroundColor(.gray)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                
                Spacer()
                
                Button(action: {
                    checkCameraPermission()
                }) {
                    HStack {
                        Image(systemName: "camera.fill")
                        Text("Take X-Ray Photo")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(10)
                }
                .padding(.horizontal)
                .padding(.bottom, 30)
            }
            .padding()
            .navigationDestination(isPresented: $showingCamera) {
                CameraView(dismiss: { showingCamera = false })
            }
            .alert("Camera Access Required", isPresented: $showingPermissionAlert) {
                Button("Settings", role: .none) {
                    if let settingsURL = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(settingsURL)
                    }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Please enable camera access in Settings to use this feature.")
            }
        }
        .navigationBarBackButtonHidden(true)
    }
    
    private func checkCameraPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            showingCamera = true
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    if granted {
                        showingCamera = true
                    } else {
                        showingPermissionAlert = true
                    }
                }
            }
        case .denied, .restricted:
            showingPermissionAlert = true
        @unknown default:
            break
        }
    }
}

#Preview {
    ContentView()
}
