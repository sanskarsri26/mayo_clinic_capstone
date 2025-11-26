Here is a polished, professional **`scripts/README.md`** you can place directly inside your `scripts/` folder.
It documents all three scripts clearly and prepares future developers or sponsors to use them confidently.

---

# 📄 **scripts/README.md**

````markdown
# Scripts Overview

This folder contains all utility scripts used for **model conversion**, **evaluation**, and **deployment
validation** for the Chest X-Ray Diagnosis App.

These scripts support:
- exporting PyTorch models to CoreML,
- verifying prediction consistency between PyTorch and CoreML,
- performing simple evaluations on test sets.

They are designed so that a future developer can easily reproduce the entire deployment pipeline.

---

# 📁 Scripts Included

## 1. `export_to_onnx.py`
Exports a trained PyTorch `.pth` model into **ONNX** format.

### **Purpose**
- Convert your PyTorch EfficientNet-V2-S checkpoint into an ONNX model.
- ONNX acts as an intermediate format for CoreML conversion.

### **Usage**
```bash
python scripts/export_to_pth.py \
    --checkpoint ml_training/models/effnetv2s_finetuned.pth \
    --output ml_training/models/effnetv2s_finetuned.mlmodel
````

### **Notes**

* The input shape is fixed at `(1, 3, 224, 224)`.
* The output will be a multi-label probability vector of size 14.

---

## 2. `pth_to_coreml.py`

Converts a PyTorch `.pth` checkpoint **directly to CoreML** `.mlmodel`.

### **Purpose**

* Load the fine-tuned EfficientNet-V2-S model.
* Trace the model with TorchScript.
* Convert it into an iOS-ready `.mlmodel` file using `coremltools`.

### **Usage**

```bash
python scripts/pth_to_coreml.py \
    --checkpoint ml_training/models/effnetv2s_finetuned.pth \
    --output ml_training/models/effnetv2s_finetuned.mlmodel
```

### **Notes**

* The script embeds the correct normalization parameters (`mean=0.5`, `std=0.5`).
* Uses CoreML "mlprogram" format for best performance on iOS 16+.
* This is the version used inside the Xcode project.

---

## 3. `compare_pth_coreml.py`

Compares predictions between the PyTorch model and the CoreML model on a test image directory.

### **Purpose**

* Validate that the CoreML model outputs match the PyTorch model.
* Detect conversion inconsistencies or preprocessing mismatches.
* Print per-class and overall differences.

### **Usage**

```bash
python scripts/compare_pth_coreml.py \
    --checkpoint ml_training/models/effnetv2s_finetuned.pth \
    --coreml ml_training/models/effnetv2s_finetuned.mlmodel \
    --images ml_training/data/test_subset_images \
    --max-samples 50
```

### **Output Includes**

* Mean absolute difference (PyTorch vs CoreML)
* Max absolute difference
* Per-class difference summary
* Counts of images evaluated

### **Notes**

* Test images must be .jpg/.jpeg/.png.
* Preprocessing matches the training pipeline (resize + normalize).


# 🏗 Folder Structure Reminder

```
scripts/
├── export_to_onnx.py
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

```

