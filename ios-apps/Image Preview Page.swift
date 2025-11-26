import SwiftUI
import Photos

struct ImagePreviewView: View {
    let image: UIImage
    let dismiss: () -> Void

    @State private var showingAnalysis = false
    @State private var showingCamera = false

    // Save-to-Photos state
    @State private var isSaving = false
    @State private var saveSucceeded = false
    @State private var saveMessage: String?
    @State private var showBanner = false

    var body: some View {
        ZStack {
            VStack(spacing: 20) {
                // Captured Image
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 500)
                    .padding()

                // Instructions / Title
                Text("Retake the image or proceed to analysis.")
                    .font(.headline)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)

                // Action Buttons
                VStack(spacing: 12) {
                    // Analyze Button
                    Button(action: { showingAnalysis = true }) {
                        HStack {
                            Image(systemName: "magnifyingglass")
                            Text("Analyze Image")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }

                    // Save to Photos (no popup)
                    Button(action: { saveToPhotos() }) {
                        HStack {
                            Image(systemName: "square.and.arrow.down")
                            Text(isSaving ? "Saving..." : "Save to Photos")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(isSaving ? Color.gray : Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    .disabled(isSaving)

                    // Retake Button
                    Button(action: {
                        showingCamera = true
                    }) {
                        HStack {
                            Image(systemName: "arrow.uturn.left")
                            Text("Retake")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.secondary.opacity(0.2))
                        .foregroundColor(.primary)
                        .cornerRadius(10)
                    }
                }
                .padding(.horizontal)
            }
            .padding()
            .navigationTitle("Preview")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarBackButtonHidden(true)
            .navigationDestination(isPresented: $showingAnalysis) {
                AnalysisView(image: image, dismiss: dismiss)
            }
            .navigationDestination(isPresented: $showingCamera) {
                CameraView(dismiss: dismiss)
            }

            // Non-modal banner/toast
            if showBanner {
                BannerView(
                    message: saveMessage ?? (saveSucceeded ? "Saved to Photos" : "Couldn’t save"),
                    isSuccess: saveSucceeded
                )
                .transition(.move(edge: .top).combined(with: .opacity))
                .zIndex(1)
                .padding(.top, 12)
                .frame(maxWidth: .infinity, alignment: .top)
            }
        }
        .animation(.spring(response: 0.35, dampingFraction: 0.85), value: showBanner)
    }
}

// MARK: - Save Helpers
private extension ImagePreviewView {
    func saveToPhotos() {
        guard !isSaving else { return }
        isSaving = true
        saveMessage = nil

        if #available(iOS 14, *) {
            PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
                switch status {
                case .authorized, .limited: self.performSave()
                default: self.finishSave(success: false, message: "Photos access denied. Enable in Settings.")
                }
            }
        } else {
            PHPhotoLibrary.requestAuthorization { status in
                switch status {
                case .authorized: self.performSave()
                default: self.finishSave(success: false, message: "Photos access denied. Enable in Settings.")
                }
            }
        }
    }

    func performSave() {
        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.creationRequestForAsset(from: self.image)
        }) { success, error in
            if let error = error {
                self.finishSave(success: false, message: error.localizedDescription)
            } else {
                self.finishSave(success: success, message: "Saved to Photos")
            }
        }
    }
    
    func finishSave(success: Bool, message: String?) {
        DispatchQueue.main.async {
            self.isSaving = false
            self.showBannerMessage(success: success, message: message ?? "")
        }
    }
    
    func showBannerMessage(success: Bool, message: String) {
        DispatchQueue.main.async {
            self.saveSucceeded = success
            self.saveMessage = message

            // Haptic feedback
            if success {
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            } else {
                UINotificationFeedbackGenerator().notificationOccurred(.warning)
            }

            // Show banner briefly
            self.showBanner = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                self.showBanner = false
            }
        }
    }
}

