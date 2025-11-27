# mayo_clinic_capstone
Capstone project for Mayo Clinic - CSE 486

Model weight link: https://drive.google.com/drive/folders/1rVv52gXj99Ue_qupf0KFz3hJa80lvchO?usp=sharing or it also located inside ml_training/models/

---

## 1. Repo Structure

```
mayo_clinic_capstone
├── README.md
├── requirements.txt       # your existing file
├── docs/
│   ├── Deployment_of_the_project.pdf
│   ├── architecture_diagrams/   # export PNGs/PDFs from your diagrams
├── ml_training/
│   ├── cheXpert_final.ipynb     # Stage 1 – DenseNet121 pretraining
│   ├── finetuning.ipynb         # Stage 2 – EfficientNet-V2-S domain adaptation
│   ├── models/
│   │   ├── densenet121_chexpert.pth.tar
│   │   ├── effnetv2s_finetuned.pth
│   └── data/
│       ├── Readme.md               # guide to download the data
├── ios-app/
│   ├── ChestXRayDiagnosisApp.xcodeproj
│   ├── ChestXRayDiagnosisApp/
│   │   ├── Models/              # .mlmodel files
│   │   ├── Views/               # SwiftUI views
│   │   ├── ViewModels/
│   │   ├── Services/            # camera, CoreML, Vision, Grad-CAM
│   │   └── Resources/           # assets, help GIF, icons
└── scripts/
    ├── export_to_pth.py
    └── eval_model.py
```

---

## 2. `How things work`

Here’s a full `README.md` you can paste into your repo and tweak names/links as needed:

````markdown
# Chest X-Ray Diagnosis App

A fully on-device iOS application for **chest X-ray analysis**, combining:

- Real-time camera capture
- Automated image quality assessment
- Lung segmentation
- Multi-label disease classification (14 CheXpert-style labels)
- Grad-CAM visual explanations

All inference runs **completely offline on device** using CoreML – no servers, no APIs.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
  - [ML Training Environment](#ml-training-environment)
  - [iOS App Environment](#ios-app-environment)
- [Model Training Pipeline](#model-training-pipeline)
- [On-Device Inference](#on-device-inference)
- [Best Practices](#best-practices)
- [Transferring Ownership](#transferring-ownership)
- [Contact](#contact)
- [License](#license)

---

## Overview

This project implements a **two-stage training + mobile deployment** workflow:

1. **Stage 1 – CheXpert Pretraining (DenseNet121)**  
   Train a DenseNet121 model on a curated subset (~20k images) of the CheXpert dataset to learn
   strong chest X-ray representations.

2. **Stage 2 – Domain Adaptation (EfficientNet-V2-S)**  
   Fine-tune an EfficientNet-V2-S model on custom low-quality subsets (white/yellow backgrounds,
   closer/further distance) to mimic phone-captured radiographs.

The final EfficientNet-V2-S model is exported to **ONNX** and then converted to **CoreML
(.mlmodel)** for deployment inside an iOS application written in **Swift 5 / SwiftUI**.

---

## Features

- **In-app camera** optimized for photographing chest X-rays
- **Lung segmentation** using a CoreML segmentation model
- **Image quality control** using Sobel + Laplacian-based sharpness metrics
- **On-device diagnosis** with EfficientNet-V2-S (14 CheXpert labels)
- **Grad-CAM heatmaps** overlaid on the user’s X-ray
- **Fully offline** – no internet or external servers required
- **iOS-native** implementation with SwiftUI, CoreML, Vision, AVFoundation

---

## System Architecture

High-level pipeline:

1. User captures a photo of a chest X-ray in the app.
2. The app runs a **lung segmentation model** (CoreML) to isolate the lungs.
3. Sharpness inside the lung region is evaluated (Sobel + Laplacian metrics).
4. If the image fails quality checks, → user is prompted to retake.
5. If it passes → the image is sent to the **EfficientNet-V2-S CoreML model** for classification.
6. Grad-CAM maps are generated and overlaid on the X-ray.
7. Final probability scores for 14 conditions are displayed.

Detailed architecture diagrams and training flows are available in  
`docs/Deployment_of_the_project.pdf`.

---

````

> **Note:** CheXpert images are *not* included due to licensing. See `ml_training/data/README.md`
> for instructions on how to request and arrange the dataset.

---

## Installation & Setup

### ML Training Environment

1. Create and activate a Conda environment:

```bash
conda create -n chexpert-env python=3.11
conda activate chexpert-env
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure you have access to a CUDA-enabled GPU (training was performed on
   the SOL supercomputer at ASU).

### iOS App Environment

1. Install **Xcode** (15.x+ recommended) on macOS.
2. Open:

```text
ios-app/ChestXRayDiagnosisApp.xcodeproj
```

3. Go to **Signing & Capabilities**:

   * Set the **Team** to your Apple Developer account.
   * Set a unique **Bundle Identifier** (e.g. `com.yourorg.ChestXRayApp`).
4. Select a physical iOS device (iOS 17.6+ recommended) and press **Run**.

---

## Model Training Pipeline

### Stage 1 – CheXpert Pretraining (DenseNet121)

Notebook: `ml_training/cheXpert_final.ipynb`

* Model: `torchvision.models.densenet121(pretrained=True)`
* Last layer replaced by `Linear(in_features, 14) + Sigmoid`
* Trained on ~20k CheXpert images with 14 labels
* Output: `densenet121_chexpert.pth.tar`

### Stage 2 – Domain Adaptation (EfficientNet-V2-S)

Notebook: `ml_training/finetuning.ipynb`

* Backbone: `torchvision.models.efficientnet_v2_s`
* Head: small MLP (fully connected layer(s) with Sigmoid) → 14 outputs
* Optimizer: **AdamW**
* Loss: **Asymmetric Loss (ASL)** for imbalanced multi-label classification
* Epochs: 40, Batch size: 12
* Uses four domain subsets:

  * `white_closerin`, `white_furtherout`
  * `yellow_closerin`, `yellow_furtherout`
* Evaluation metric: **AUC** (per label and aggregate)

### Export to CoreML

1. Export PyTorch → ONNX:

```bash
python scripts/export_to_onnx.py
```

2. Convert ONNX → CoreML:

```bash
python scripts/convert_onnx_to_coreml.py
```

3. Place the resulting `.mlmodel` file into:

```text
ios-app/ChestXRayDiagnosisApp/Models/
```

---

## On-Device Inference

* Runtime: **CoreML** + **Vision** frameworks.
* Model input: `(1, 3, 224, 224)` RGB image, normalized with mean `[0.5, 0.5, 0.5]`
  and std `[0.5, 0.5, 0.5]`.
* Output: 14-dimensional probability vector in `[0, 1]`.
* Grad-CAM maps are computed and overlaid on the X-ray inside the app.

---

## Best Practices

* Keep **OpenCV2** and C++ modules up to date.
* Maintain Swift 5 compatibility; update via Xcode migration tools.
* Target **iOS 17.6+** for consistent CoreML & Vision behavior.
* Match on-device preprocessing **exactly** to the training transforms.
* Log training runs (AUC, loss curves) and keep checkpoints versioned.
* Use GitHub branching and tags to track releases.

A detailed best-practices document is included in
`docs/Deployment_of_the_project.pdf`.

---

## Transferring Ownership

To hand this project over to a sponsor or new team:

1. Share this GitHub repository and ensure they have **read/write access**.
2. Provide:

   * Access to trained model files (or instructions to retrain).
   * Instructions for requesting the CheXpert dataset.
   * Access to any Apple Developer accounts used for signing.
3. Have the sponsor:

   * Update Xcode Signing & Capabilities with their own Team + Bundle ID.
   * Confirm they can build & run the app on an iOS device.

An approval/signature page is included at the end of
`docs/Deployment_of_the_project.pdf`.

---

## Contact

For questions, maintenance, or future development:

* **Connor McMahon** – `mcmahon.connor.04@gmail.com`
* **Sanskar Srivastava** – `ssriva94@asu.edu`

---

## License

> **Note:** This repository currently has no explicit open-source license.
> By default, this means *all rights reserved*.

CheXpert dataset usage is governed by Stanford’s dataset-specific license and
is intended for **non-commercial research** purposes only.

```
