//
//  HelpOverlay.swift
//  ChestX-RayImaging
//
//  Created by Connor McMahon on 10/29/25.
//

import SwiftUI

struct HelpOverlay: View {
    var onDismiss: () -> Void

    var body: some View {
        ZStack {
            Color.black.opacity(0.55)
                .ignoresSafeArea()

            VStack(spacing: 16) {
                Text("How to position your camera")
                    .font(.headline)
                    .multilineTextAlignment(.center)

                // The GIF
                GIFView(name: "camera-help")
                    .frame(height: 260)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                // Quick tips (edit to taste)
                VStack(alignment: .leading, spacing: 8) {
                    Label("Fill the frame with the chest — include both lungs", systemImage: "rectangle.dashed.badge.record")
                    Label("Hold steady for 1–2 seconds", systemImage: "hand.raised")
                    Label("Good light, no shadows or reflections", systemImage: "sun.max")
                }
                .font(.subheadline)
                .frame(maxWidth: .infinity, alignment: .leading)

                Button(action: onDismiss) {
                    Text("Got it")
                        .bold()
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
            }
            .padding(20)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
            .padding(.horizontal, 24)
        }
    }
}
