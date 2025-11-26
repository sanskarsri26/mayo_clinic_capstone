//
//  GIFView.swift
//  ChestX-RayImaging
//
//  Created by Connor McMahon on 10/29/25.
//

import SwiftUI
import WebKit

struct GIFView: UIViewRepresentable {
    let name: String   // without ".gif"

    func makeUIView(context: Context) -> WKWebView {
        let web = WKWebView()
        web.isOpaque = false
        web.backgroundColor = .clear
        web.scrollView.isScrollEnabled = false
        web.scrollView.bounces = false
        web.isUserInteractionEnabled = false

        if let url = Bundle.main.url(forResource: name, withExtension: "gif") {
            // Load file URL directly so it loops natively
            web.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
            // Force the GIF to scale to fit container
            let js = """
            var meta = document.createElement('meta');
            meta.setAttribute('name','viewport');
            meta.setAttribute('content','width=device-width, initial-scale=1.0, maximum-scale=1.0');
            document.getElementsByTagName('head')[0].appendChild(meta);
            document.body.style.margin='0'; document.documentElement.style.margin='0';
            """
            web.evaluateJavaScript(js, completionHandler: nil)
        }
        return web
    }

    func updateUIView(_ uiView: WKWebView, context: Context) { }
}
