#!/usr/bin/env python3
"""
Compare PyTorch (.pth) and CoreML (.mlmodel) outputs on a test subset of images.

Usage (from repo root):

    python scripts/compare_pth_coreml.py \
        --checkpoint ml_training/models/effnetv2s_finetuned.pth \
        --coreml ml_training/models/effnetv2s_finetuned.mlmodel \
        --images path/to/test_images \
        --max-samples 100

This will:
- Load the PyTorch model (EffNetV2SModel)
- Load the CoreML model
- Run both on up to N images from --images
- Print statistics about differences between outputs
"""

import argparse
import os
import glob
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision import transforms
from PIL import Image

import coremltools as ct


# -------------------------
# 1. Model definition (same as in pth_to_coreml.py)
# -------------------------

class EffNetV2SModel(nn.Module):
    def __init__(self, num_classes: int = 14):
        super().__init__()
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.backbone(x)
        probs = self.sigmoid(logits)
        return probs


def load_pytorch_model(checkpoint_path: str, num_classes: int = 14) -> nn.Module:
    device = torch.device("cpu")
    model = EffNetV2SModel(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "state_dict" not in state:
        model.load_state_dict(state)
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        raise RuntimeError("Unexpected checkpoint format in load_pytorch_model()")

    model.to(device)
    model.eval()
    return model


def load_coreml_model(coreml_path: str):
    print(f"Loading CoreML model from {coreml_path}")
    mlmodel = ct.models.MLModel(coreml_path)
    return mlmodel


# -------------------------
# 2. Preprocessing (must match training)
# -------------------------

def get_preprocess_transform(image_size: int = 224):
    """
    Training used:
      - Resize to 224x224
      - ToFloat(max=255)
      - Normalize(mean=0.5, std=0.5)

    In torchvision terms:
      - transforms.Resize
      - transforms.ToTensor
      - transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])


# -------------------------
# 3. Comparison logic
# -------------------------

def run_comparison(
    checkpoint_path: str,
    coreml_path: str,
    images_dir: str,
    max_samples: int = 100,
    image_size: int = 224,
    num_classes: int = 14,
):
    device = torch.device("cpu")
    torch_model = load_pytorch_model(checkpoint_path, num_classes=num_classes)
    coreml_model = load_coreml_model(coreml_path)
    preprocess = get_preprocess_transform(image_size=image_size)

    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        print(f"No images found in {images_dir} (supported: png, jpg, jpeg).")
        return

    if max_samples is not None and max_samples > 0:
        image_paths = image_paths[:max_samples]

    diffs = []

    print(f"Comparing on {len(image_paths)} images...")

    for img_path in tqdm(image_paths):
        img = Image.open(img_path).convert("RGB")

        # PyTorch inference
        tensor = preprocess(img).unsqueeze(0).to(device)  # shape (1,3,H,W)
        with torch.no_grad():
            pt_out = torch_model(tensor).cpu().numpy().reshape(-1)

        # CoreML inference
        # Because we set the input as ImageType in conversion, we can pass the PIL image directly
        coreml_out_dict = coreml_model.predict({"input": img})
        # Find the first output key (you can also hardcode if you know the name)
        out_key = list(coreml_out_dict.keys())[0]
        cm_out = np.array(coreml_out_dict[out_key]).reshape(-1)

        if pt_out.shape != cm_out.shape:
            print(f"Shape mismatch for {img_path}: pt={pt_out.shape}, coreml={cm_out.shape}")
            continue

        diff = np.abs(pt_out - cm_out)
        diffs.append(diff)

    if not diffs:
        print("No valid comparisons made (diff list is empty).")
        return

    diffs = np.stack(diffs, axis=0)  # (N, num_classes)

    mean_abs_diff_per_class = diffs.mean(axis=0)
    max_abs_diff_per_class = diffs.max(axis=0)
    overall_mean = diffs.mean()
    overall_max = diffs.max()

    print("\n=== Comparison Summary ===")
    print(f"Number of samples: {len(image_paths)}")
    print(f"Number of classes: {num_classes}")
    print(f"Overall mean |PyTorch - CoreML|: {overall_mean:.6f}")
    print(f"Overall max  |PyTorch - CoreML|: {overall_max:.6f}\n")

    print("Per-class mean absolute difference:")
    for i, v in enumerate(mean_abs_diff_per_class):
        print(f"  Class {i:02d}: {v:.6f}")

    print("\nPer-class max absolute difference:")
    for i, v in enumerate(max_abs_diff_per_class):
        print(f"  Class {i:02d}: {v:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Compare .pth model vs .mlmodel on test images")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the PyTorch checkpoint (.pth)")
    parser.add_argument("--coreml", type=str, required=True,
                        help="Path to the CoreML model (.mlmodel)")
    parser.add_argument("--images", type=str, required=True,
                        help="Directory containing test images (jpg/png)")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum number of images to evaluate (0 = all)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Resize images to this size for evaluation")
    parser.add_argument("--num-classes", type=int, default=14,
                        help="Number of output classes")
    args = parser.parse_args()

    if args.max_samples == 0:
        args.max_samples = None

    run_comparison(
        checkpoint_path=args.checkpoint,
        coreml_path=args.coreml,
        images_dir=args.images,
        max_samples=args.max_samples,
        image_size=args.image_size,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
