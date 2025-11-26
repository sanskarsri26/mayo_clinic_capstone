//
//  CustomCameraView.swift
//  ChestX-RayImaging
//
//  Created by Connor McMahon on 3/20/25.
//

import SwiftUI
import AVFoundation

struct CustomCameraView: View {
    @Binding var capturedImage: UIImage?
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        CameraView(dismiss: { dismiss() })
    }
}

#Preview {
    CustomCameraView(capturedImage: .constant(nil))
}
