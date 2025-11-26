````markdown
# Models Directory

This folder stores all **trained model artifacts** used in the Chest X-Ray Diagnosis App, along with
intermediate export formats for deployment (PyTorch → ONNX → CoreML).

Because some of these files may be large, they may be tracked with **Git LFS** or provided via
external storage if needed.

---

## 1. File Overview

Typical contents of this folder:

- `densenet121_chexpert.pth.tar`  
- `effnetv2s_finetuned.pth`  
- `effnetv2s_finetuned.onnx`  
- `effnetv2s_finetuned.mlmodel`  

Your exact filenames may differ slightly, but the roles are as follows.

---

## 2. Model Files and Their Roles

### 2.1 `densenet121_chexpert.pth.tar`
**Type:** PyTorch checkpoint  
**Stage:** Stage 1 – CheXpert Pretraining  

- Trained on ~20k images from the CheXpert dataset.
- Uses DenseNet121 backbone with a 14-label multi-label classifier.
- Purpose:
  - Provides a strong chest X-ray feature representation.
  - Serves as the CheXpert-pretrained base used for domain understanding.

This model is typically **not deployed** to iOS directly, but is important historically and for
retraining/improvement workflows.

---

### 2.2 `effnetv2s_finetuned.pth`
**Type:** PyTorch checkpoint  
**Stage:** Stage 2 – Domain Adaptation (Deployment model in training form)

- Architecture: EfficientNet-V2-S + MLP head for 14-label multi-label classification.
- Trained on:
  - High-quality CheXpert subset
  - Domain-specific subsets (white/yellow, closer/further) to simulate phone captures
- Uses Asymmetric Loss (ASL) and AdamW optimizer.
- This is the **source model** for all export steps (ONNX, CoreML).

If you want to continue training or fine-tuning, this is the checkpoint to start from.

---

### 2.3 `effnetv2s_finetuned.onnx`
**Type:** ONNX model  
**Stage:** Intermediate deployment format

- Exported from `effnetv2s_finetuned.pth` using `export_to_onnx.py`.
- Used as the input to CoreML conversion via `coremltools`.
- Allows:
  - Cross-framework compatibility
  - Quick validation using ONNX Runtime

If you update the PyTorch model, you must regenerate this ONNX file.

---

### 2.4 `effnetv2s_finetuned.mlmodel`
**Type:** CoreML model  
**Stage:** Final deployment artifact for iOS

- Used directly inside the iOS app.
- Placed (or copied) into:

  ```text
  ios-app/ChestXRayDiagnosisApp/Models/
````

* Xcode compiles this into `.mlmodelc` at build time.
* All on-device inference in the app uses this model.

If the sponsor or a developer wants to update the app’s predictions, this is the file they replace
after retraining + conversion.

---

## 3. Regenerating Models

If you need to retrain or regenerate any model files:

1. **Retrain Stage 1 (DenseNet121)**

   * Use `cheXpert_final.ipynb`
   * Output: `densenet121_chexpert.pth.tar`

2. **Retrain Stage 2 (EfficientNet-V2-S)**

   * Use `finetuning.ipynb`
   * Output: `effnetv2s_finetuned.pth`

3. **Export to ONNX**

   * Run:

     ```bash
     python ../scripts/export_to_onnx.py
     ```
   * Output: `effnetv2s_finetuned.onnx`

4. **Convert ONNX → CoreML**

   * Run:

     ```bash
     python ../scripts/convert_onnx_to_coreml.py
     ```
   * Output: `effnetv2s_finetuned.mlmodel`

Then copy the updated `.mlmodel` into the `ios-app` project.

---

## 4. Storage & Versioning

* Due to size, consider using **Git LFS** for:

  * `.pth` / `.tar` files
  * `.onnx` files
  * `.mlmodel` files (if large)
* Tag important model versions with Git tags (e.g., `v1.0-model-release`).
* Document major changes (architecture, dataset version, hyperparameters) in `docs/model_notes.md`.

---

## 5. Security & Licensing Notes

* Model weights are derived from the **CheXpert dataset**, which is governed by a
  non-commercial research license.
* Before using these models in clinical, commercial, or production settings, ensure:

  * License compliance with CheXpert
  * Proper validation and regulatory approvals (if applicable)

---

## 6. Contact

For questions about the models, training, or deployment:

**Connor McMahon**
📧 `mcmahon.connor.04@gmail.com`

**Sanskar Srivastava**
📧 `ssriva94@asu.edu`

```

