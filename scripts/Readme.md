````markdown
# Scripts Overview

This folder contains all utility scripts used for **model conversion**, **evaluation**, and
**deployment validation** for the Chest X-Ray Diagnosis App.

These scripts support:
- converting PyTorch `.pth` models directly to CoreML `.mlmodel`,
- verifying prediction consistency between PyTorch and CoreML,
- performing evaluation on a test image subset.

They are designed so that future developers can easily reproduce the entire deployment pipeline.

---

# 📁 Scripts Included

## 1. `pth_to_coreml.py`

Converts a trained PyTorch `.pth` checkpoint **directly into a CoreML `.mlmodel`** ready for iOS deployment.

### **Purpose**
- Load the fine-tuned EfficientNet-V2-S model from the `.pth` checkpoint.
- Trace the model using TorchScript.
- Convert it to CoreML using `coremltools`.
- Save the resulting `.mlmodel` for use inside the iOS app.

### **Usage**
```bash
python scripts/pth_to_coreml.py \
    --checkpoint ml_training/models/effnetv2s_finetuned.pth \
    --output ml_training/models/effnetv2s_finetuned.mlmodel
````

### **Notes**

* Embeds correct preprocessing (`mean=0.5`, `std=0.5`) into the CoreML model.
* Exports to the **ML Program** format recommended for iOS 16+.
* The resulting `.mlmodel` should be copied into:

  ```
  ios-app/ChestXRayDiagnosisApp/Models/
  ```

---

## 2. `compare_pth_coreml.py`

Compares outputs from the PyTorch `.pth` model and the CoreML `.mlmodel` to ensure the conversion was correct.

### **Purpose**

* Run inference on both models using the same test images.
* Detect inconsistencies due to:

  * conversion errors,
  * preprocessing mismatches,
  * floating-point differences.
* Print per-class and overall prediction differences.

### **Usage**

```bash
python scripts/compare_pth_coreml.py \
    --checkpoint ml_training/models/effnetv2s_finetuned.pth \
    --coreml ml_training/models/effnetv2s_finetuned.mlmodel \
    --images ml_training/data/test_subset_images \
    --max-samples 50
```

### **Output Includes**

* Overall mean absolute difference
* Overall maximum difference
* Per-class differences (0–13)
* Number of evaluated images

### **Notes**

* Test images must be `.jpg`, `.jpeg`, or `.png`.
* Uses the **exact same preprocessing** as training:

  * resize → 224×224
  * ToTensor
  * normalize (mean = std = 0.5)

---

# 🏗 Folder Structure Reminder

```
scripts/
├── pth_to_coreml.py
└── compare_pth_coreml.py
```

---

# 📞 Contact

For help with model conversion or script updates:

**Connor McMahon**
📧 `mcmahon.connor.04@gmail.com`

**Sanskar Srivastava**
📧 `ssriva94@asu.edu`

---

```
