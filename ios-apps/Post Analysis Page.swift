import SwiftUI

struct AnalysisView: View {
    let image: UIImage
    let dismiss: () -> Void
    @State private var showingSaveAlert = false
    @State private var showingCamera = false
    @State private var gradCAMImage: UIImage?
    @State private var isLoading = true
    @State private var predictions: [String: Float] = [:]
    @State private var imageHeight: CGFloat = 300  // Store image height
    
    private let modelService = ModelService()
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Captured Image
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: imageHeight)
                    .padding()
                
                if isLoading {
                    ProgressView("Analyzing image...")
                        .padding()
                } else {
                    // Grad-CAM Visualization
                    if let gradCAMImage = gradCAMImage {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Model Attention Map")
                                .font(.headline)
                            
                            Image(uiImage: gradCAMImage)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxHeight: imageHeight)  // Match original image height
                                .cornerRadius(10)
                            
                            // Color Legend
                            HStack {
                                Text("Low")
                                Spacer()
                                ForEach([
                                    Color(red: 0, green: 0, blue: 1),    // Blue
                                    Color(red: 0, green: 1, blue: 1),    // Cyan
                                    Color(red: 0, green: 1, blue: 0),    // Green
                                    Color(red: 1, green: 1, blue: 0),    // Yellow
                                    Color(red: 1, green: 0, blue: 0)     // Red
                                ], id: \.self) { color in
                                    Rectangle()
                                        .fill(color)
                                        .frame(width: 20, height: 20)
                                }
                                Spacer()
                                Text("High")
                            }
                            .padding(.horizontal)
                        }
                        .padding()
                        .background(Color(.systemBackground))
                        .cornerRadius(15)
                        .shadow(radius: 5)
                    }
                    
                    // Detected Condition
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Detected Conditions")
                            .font(.headline)
                            .padding(.bottom, 4)
                        
                        if predictions.isEmpty {
                            Text("No significant conditions detected")
                                .foregroundColor(.secondary)
                        } else {
                            // Filter and sort conditions with > 50% probability
                            let significantConditions = predictions
                                //.filter { $0.value > 0.5 }
                                .sorted { $0.value > $1.value }
                            
                            if significantConditions.isEmpty {
                                Text("No high-probability conditions detected")
                                    .foregroundColor(.secondary)
                            } else {
                                ForEach(significantConditions, id: \.key) { condition, probability in
                                    HStack {
                                        Text(condition)
                                            .font(.title3)
                                            .bold()
                                        Spacer()
                                        Text(String(format: "%.1f%%", probability * 100))
                                            .font(.title3)
                                            .bold()
                                            .foregroundColor(confidenceColor(for: probability))
                                    }
                                    .padding(.vertical, 4)
                                }
                            }
                        }
                    }
                    .padding()
                    .background(Color(.systemBackground))
                    .cornerRadius(15)
                    .shadow(radius: 5)
                }
                
                // Action Buttons
                HStack(spacing: 20) {
                    Button(action: {
                        showingCamera = true
                    }) {
                        HStack {
                            Image(systemName: "arrow.counterclockwise")
                            Text("Retake")
                        }
                        .frame(width: 150)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    
                    Button(action: {
                        showingSaveAlert = true
                    }) {
                        HStack {
                            Image(systemName: "house.fill")
                            Text("Return Home")
                        }
                        .frame(width: 150)
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                }
                .padding(.horizontal)
            }
            .padding()
        }
        .navigationTitle("Analysis Results")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(true)
        .alert("Return to Home?", isPresented: $showingSaveAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Return", role: .none) {
                dismiss()
            }
        } message: {
            Text("This will discard the current analysis results.")
        }
        .navigationDestination(isPresented: $showingCamera) {
            CameraView(dismiss: dismiss)
        }
        .onAppear {
            analyzeImage()
        }
    }
    
    private func confidenceColor(for value: Float) -> Color {
        switch value {
        case 0.0..<0.3:
            return .red
        case 0.3..<0.5:
            return .orange
        case 0.5..<0.7:
            return .yellow
        default:
            return .green
        }
    }
    
    private func confidenceLabel(for value: Float) -> String {
        switch value {
        case 0.0..<0.3:
            return "Low Confidence"
        case 0.3..<0.5:
            return "Moderate Confidence"
        case 0.5..<0.7:
            return "High Confidence"
        default:
            return "Very High Confidence"
        }
    }
    
    private func analyzeImage() {
        modelService.analyzeImage(image) { probability, visualization, predictions in
            DispatchQueue.main.async {
                self.gradCAMImage = visualization
                self.predictions = predictions
                self.isLoading = false
            }
        }
    }
} 
