//
//  ChestX_RayImagingApp.swift
//  ChestX-RayImaging
//
//  Created by Connor McMahon on 3/20/25.
//

import SwiftUI

@main
struct ChestX_RayImagingApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
