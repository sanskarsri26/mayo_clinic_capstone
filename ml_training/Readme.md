
```markdown
# Machine Learning Training Pipeline

This folder contains all notebooks, scripts, and model files required to train, fine-tune, evaluate, and
export the machine learning models used in the **Chest X-Ray Diagnosis App**.  
It includes both stages of the training pipeline—CheXpert pretraining and domain-adapted fine-tuning—
along with conversion utilities for ONNX and CoreML deployment.

---

# 1. Overview

The machine learning system follows a **two-stage training pipeline**:

### **Stage 1 — CheXpert Pretraining (DenseNet121)**
A DenseNet121 model is trained on a curated subset of the CheXpert dataset (~20k images).  
This stage enables the model to learn strong chest radiology representations.

### **Stage 2 — Domain Adaptation (EfficientNet-V2-S)**
A separate EfficientNet-V2-S model is fine-tuned on low-quality domain-specific subsets mimicking
phone-captured X-rays (white/yellow backgrounds, close/far captures).

The final EfficientNet-V2-S model is exported to ONNX and then converted to CoreML for on-device
inference in the iOS application.

---

# 2. Repository Structure

```

ml_training/
├── cheXpert_final.ipynb         # Stage 1 – DenseNet121 CheXpert pretraining
├── finetuning.ipynb             # Stage 2 – EffNetV2-S domain adaptation
├── models/
│   ├── densenet121_chexpert.pth.tar
│   ├── effnetv2s_finetuned.pth
│   └── effnetv2s_finetuned.mlmodel
├── data/
│   ├── README.md                # instructions for obtaining CheXpert dataset
│   ├── chexpert_csvs/           # TRAIN_CSV, TEST_A_CSV, TEST_B_CSV
│   └── subsets/                 # white/yellow closer/further domain subsets
└── scripts/
├── export_to_onnx.py
├── convert_onnx_to_coreml.py
└── eval_model.py

```

---

# 3. Stage 1 — CheXpert Pretraining (DenseNet121)

**Notebook:** `cheXpert_final.ipynb`  
This notebook trains a DenseNet121 model on the CheXpert subset.

### **Model**
- `torchvision.models.densenet121(pretrained=True)`
- Final layer replaced with:
```

Linear(in_features, 14)
Sigmoid()

```

### **Output**
- `models/densenet121_chexpert.pth.tar`

This checkpoint is used as the "CheXpert awareness" baseline for the second stage.

---

# 4. Stage 2 — Domain Adaptation (EfficientNet-V2-S)

**Notebook:** `finetuning.ipynb`  
Fine-tunes EfficientNet-V2-S under four domain shift scenarios.

### **Model**
- Backbone: `efficientnet_v2_s`  
- MLP head for classification:
```

Linear(in_features → 14)
Sigmoid()

```

### **Domain Subsets**
Located in `data/subsets/`:
- `white_closerin/`
- `white_furtherout/`
- `yellow_closerin/`
- `yellow_furtherout/`

These subsets imitate real-world phone capture scenarios.

### **Training Details**
- Loss: **Asymmetric Loss (ASL)**
- Optimizer: **AdamW**
- Epochs: 40
- Batch size: 12
- Mixed precision: **enabled**
- No freezing — entire network fine-tuned

### **Augmentations**
Using Albumentations:
- RandomResizedCrop
- HorizontalFlip
- ColorJitter
- RandomBrightnessContrast
- Gaussian Noise

### **Output**
- `models/effnetv2s_finetuned.pth`

---

# 5. Preprocessing (Training & Inference)

To ensure accuracy, training and iOS preprocessing must match exactly.

### **Image Preprocessing**
- Resize to **224×224**
- Convert to float
- Normalize using:
```

mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]

````

### **On-Device Matching**
The same transformation is applied inside the iOS CoreML pipeline.

---

# 6. Model Conversion

### **Step 1 — Export PyTorch → ONNX**
`models/effnetv2s_finetuned.pth → models/effnetv2s_finetuned.onnx`

Run:
```bash
python scripts/export_to_onnx.py
````

### **Step 2 — Convert ONNX → CoreML**

```bash
python scripts/convert_onnx_to_coreml.py
```

### **Step 3 — Integrate into iOS App**

Move the `.mlmodel` file into:

```
ios-app/ChestXRayDiagnosisApp/Models/
```

---

# 7. Evaluation

### **Metrics**

* Primary: AUC (per label + macro average)
* Validation set: Custom split from CheXpert subset
* Outputs: ROC curves, classification probabilities, per-label performance

### **Script**

Use `scripts/eval_model.py` for additional evaluation after training.

---

# 8. Training Requirements

### **Environment**

* Python 3.11
* Conda environment recommended
* CUDA GPU required (training completed on ASU’s SOL supercomputer)

### **Dependencies**

See root-level `requirements.txt`.

---

# 9. Dataset Access

The CheXpert dataset cannot be redistributed.

Instructions for requesting and preparing the dataset are provided in:
`ml_training/data/README.md`

---

# 10. Contact

For additional help with retraining or modifying the ML models:

**Connor McMahon**
📧 *[mcmahon.connor.04@gmail.com](mailto:mcmahon.connor.04@gmail.com)*

**Sanskar Srivastava**
📧 *[ssriva94@asu.edu](mailto:ssriva94@asu.edu)*

---

This folder contains everything needed to fully reproduce or extend the training pipeline
for the Chest X-Ray Diagnosis App.

```
