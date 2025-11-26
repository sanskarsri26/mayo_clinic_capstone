# Documentation Overview

This folder contains all project documentation required for understanding, maintaining, and extending the **Chest X-Ray Diagnosis App**.  
It includes architecture diagrams, training notes, deployment instructions, and supporting materials for both the machine learning pipeline and the iOS application.

## 📂 Contents

### 1. `Deployment_of_the_project.pdf`
This is the **complete deployment document** handed off to the project sponsor. It includes:

- Title page (team, sponsor, course)
- Technical documentation  
  - Architecture diagrams  
  - File navigation  
  - Installation process  
  - Licenses  
  - Libraries & dependencies  
  - API/SDK details  
  - Full machine learning training pipeline  
- Best practices  
- User guide  
- Ownership transfer information  
- Sponsor signature page  

This PDF is the **primary reference** for fully understanding the entire system workflow.

---

### 2. `architecture_diagrams/`
This folder contains exported diagrams used throughout the documentation, including:

- High-level system pipeline  
- Training pipeline flowchart  
- On-device inference sequence diagram  
- Model-level architecture diagrams  
- Data flow & preprocessing diagrams (if applicable)  

These diagrams help visualize how the ML models, iOS app, preprocessing, and inference components work together.

---

### 3. `model_notes.md` (Optional)
Provides additional notes about:

- DenseNet121 CheXpert pretraining (Stage 1)
- EfficientNet-V2-S domain adaptation (Stage 2)
- Model head (MLP) explanation
- Data subsets and preprocessing consistency
- Conversion steps (PyTorch → ONNX → CoreML)

This file serves as a **developer-friendly reference** when retraining or modifying the models.

---

## 🧠 How to Use This Folder

This directory is intended for:

### ✔ Sponsors  
To understand the full deployment, usage, and ownership of the project.

### ✔ Developers  
To quickly onboard and begin working with:
- The ML pipeline  
- The iOS app  
- The file structure  
- The conversion tools  

### ✔ Future Maintainers  
To keep the project updated through:
- Model retraining  
- Dependency upgrades  
- iOS SDK migrations  
- Hardware/OS compatibility fixes  

---

## 📌 Notes

- No dataset images are included due to CheXpert licensing restrictions.
- Model files (`.pth`, `.onnx`, `.mlmodel`) are stored separately in the `ml_training/models/` directory.
- If additional documentation is created (test reports, UI mockups, etc.), place them here.

---

## 📞 Contact Information

For questions or further assistance:

**Connor McMahon**  
📧 *mcmahon.connor.04@gmail.com*

**Sanskar Srivastava**  
📧 *ssriva94@asu.edu*

---

This folder ensures the long-term maintainability and transparency of the entire project.  
Please keep all future documentation updates inside this directory.

