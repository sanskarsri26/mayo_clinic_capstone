```markdown
# Model Notes

This document provides a detailed explanation of the machine learning models, training procedures,
data processing steps, and conversion workflows used in the **Chest X-Ray Diagnosis App**.
It serves as a reference for future developers, maintainers, and sponsors who may retrain,
improve, or extend the current models.

---

# 1. Overview of the Machine Learning Pipeline

The ML pipeline consists of **two major training stages**, followed by model export and deployment:

### **Stage 1 — CheXpert Pretraining (DenseNet121)**
A DenseNet121 model is trained on a curated ~20k subset of the CheXpert dataset to learn
radiology-specific chest X-ray features.  
This model provides a strong medical initialization for the second stage.

### **Stage 2 — Domain Adaptation (EfficientNet-V2-S)**
A separate EfficientNet-V2-S model is fine-tuned on custom low-quality subsets
(white/yellow backgrounds, farther/closer distances) to simulate phone-captured X-ray images.

### **Deployment — ONNX → CoreML Conversion**
The final EfficientNet-V2-S model is exported to ONNX, converted to CoreML, and integrated into
the iOS application for on-device inference.

---

# 2. Stage 1 — CheXpert Pretraining (DenseNet121)

### **Purpose**
- Learn foundational radiographic features
- Provide domain awareness for the second-stage model
- Train on high-quality, consistent medical imaging data

### **Model**
- `torchvision.models.densenet121(pretrained=True)`
- Final classification layer replaced with:
```

Linear(in_features, 14)
Sigmoid()

```

### **Training Details**
- Dataset: ~20,000 high-quality CheXpert images
- Labels: 14 multi-label chest findings
- Transformations: Resize → Normalize
- Loss: BCEWithLogits or ASL variant
- Optimizer: AdamW
- Outputs:
- `densenet121_chexpert.pth.tar` (checkpoint)
- Validation AUC logs

This checkpoint is **not directly deployed**, but influences the training strategy and feature behavior of the second-stage model.

---

# 3. Stage 2 — Domain Adaptation (EfficientNet-V2-S)

### **Purpose**
Train the final deployment model to handle **real-world, low-quality phone images**, including:
- white or yellow X-ray backgrounds
- varying distances from screen
- lighting inconsistencies
- mild noise or blur

### **Model**
- `torchvision.models.efficientnet_v2_s`
- Replaces classification layer with custom **MLP head**:
```

Linear(in_features → 14)
Sigmoid()

````

### ❗ What is the MLP Head?
The **MLP head** is the final classification layer (or layers) added on top of a backbone model.

It consists of:
- Fully connected (FC) layer(s)
- Activation function (Sigmoid for multi-label classification)

**It does NOT include the loss function.**  
Loss is applied separately during training.

### **Training Details**
- Epochs: 40  
- Batch Size: 12  
- Loss: Asymmetric Loss (ASL) — best for imbalanced multi-label data  
- Optimizer: AdamW  
- Precision: Mixed precision training  
- No layer freezing — entire model fine-tunes  

### **Dataset (Domain Subsets)**
Four subsets created to mimic phone capture variations:

- `white_closerin`
- `white_furtherout`
- `yellow_closerin`
- `yellow_furtherout`

### **Augmentations**
Using Albumentations:
- RandomResizedCrop
- HorizontalFlip
- ColorJitter
- RandomBrightnessContrast
- Gaussian noise
- Normalization

### **Outputs**
- `effnetv2s_finetuned.pth`
- Training metrics (AUC curves)
- Final ONNX + CoreML exports

---

# 4. Preprocessing Pipeline (Training & On-Device)

To maintain consistency between training and iOS inference:

### **Training Preprocessing**
- Resize to **224×224**
- Convert to float
- Normalize to mean `[0.5, 0.5, 0.5]` and std `[0.5, 0.5, 0.5]`

### **iOS App Preprocessing**
Implemented in Swift/CoreMLVision:

- Image resized to **224×224**
- Pixel buffer normalized to the same values as training
- Correct RGB channel order ensured

This consistency is crucial — mismatched transforms can degrade accuracy by 20–40%.

---

# 5. Evaluation and Metrics

- Primary metric: **AUC (Area Under ROC Curve)**  
- Secondary metrics: per-label AUC, training/validation loss curves  
- Validation set: custom split derived from CheXpert subset

The final model meets acceptable performance for on-device multiclass chest X-ray analysis.

---

# 6. ONNX and CoreML Conversion

### **Step 1 — Export PyTorch → ONNX**
Script example:
```bash
python scripts/export_to_onnx.py
````

### **Step 2 — Convert ONNX → CoreML using coremltools**

```bash
python scripts/convert_onnx_to_coreml.py
```

### **Step 3 — Integrate in iOS**

Place the resulting `.mlmodel` file into:

```
ios-app/ChestXRayDiagnosisApp/Models/
```

Xcode automatically handles:

* Model compilation to `.mlmodelc`
* Compute unit selection (CPU/GPU/ANE)
* Versioning

---

# 7. Model Files Summary

| File                           | Purpose                           |
| ------------------------------ | --------------------------------- |
| `densenet121_chexpert.pth.tar` | Stage 1 checkpoint (pretraining)  |
| `effnetv2s_finetuned.pth`      | Final trained PyTorch model       |
| `effnetv2s_finetuned.onnx`     | Export for CoreML conversion      |
| `effnetv2s_finetuned.mlmodel`  | CoreML inference model            |
| `effnetv2s_finetuned.mlmodelc` | Compiled CoreML model used by iOS |

---

# 8. Future Model Improvements (Optional Ideas)

* Quantization (float16, int8) to boost speed and reduce size
* Pruning to reduce inference time on older iPhones
* Incorporating image enhancement preprocessing
* Expanding dataset with synthetic phone-captured variations
* Using Vision Transformers (ViT) for improved feature attribution

---

# 9. Contact

For retraining or modifying the models:

**Connor McMahon**
📧 [mcmahon.connor.04@gmail.com](mailto:mcmahon.connor.04@gmail.com)

**Sanskar Srivastava**
📧 [ssriva94@asu.edu](mailto:ssriva94@asu.edu)

---

This document ensures reproducibility, maintainability, and clarity for anyone interacting with or extending the ML models powering the Chest X-Ray Diagnosis App.

```
