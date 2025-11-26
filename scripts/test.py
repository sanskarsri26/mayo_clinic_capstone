

import argparse
import os
import sys

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import coremltools as ct


# -------------------------
# 1. Model definition
# -------------------------

class EffNetV2SModel(nn.Module):
    """
    Wrap EfficientNet-V2-S with a custom classification head for 14 labels
    and a Sigmoid activation for multi-label classification.

    IMPORTANT:
    Make sure this matches exactly the architecture you used in finetuning.ipynb.
    Adjust num_classes, head, or activation if needed.
    """

    def __init__(self, num_classes: int = 14):
        super().__init__()
        # Load ImageNet-pretrained backbone
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        # Final activation for multi-label
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.backbone(x)
        probs = self.sigmoid(logits)
        return probs


# -------------------------
# 2. Conversion logic
# -------------------------

def load_model(checkpoint_path: str, num_classes: int = 14) -> nn.Module:
    device = torch.device("cpu")
    model = EffNetV2SModel(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device)

    # If you saved with torch.save(model.state_dict())
    if isinstance(state, dict) and "state_dict" not in state:
        model.load_state_dict(state)
    # If you saved {"state_dict": ...}
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        raise RuntimeError(
            "Unexpected checkpoint format. "
            "Update load logic in load_model() accordingly."
        )

    model.to(device)
    model.eval()
    return model


def convert_to_coreml(
    model: nn.Module,
    output_path: str,
    image_size: int = 224,
    num_channels: int = 3,
):
    # Example input for tracing
    example_input = torch.randn(1, num_channels, image_size, image_size)

    # Trace to TorchScript
    traced_model = torch.jit.trace(model, example_input)

    # NOTE: training normalization is: ToFloat + Normalize(mean=0.5, std=0.5)
    # That is equivalent to: scale = 1/127.5, bias = [-1, -1, -1]
    scale = 1.0 / 127.5
    bias = [-1.0, -1.0, -1.0]

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="input",
                shape=example_input.shape,
                scale=scale,
                bias=bias,
            )
        ],
        convert_to="mlprogram",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    print(f"Saved CoreML model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert .pth checkpoint to CoreML .mlmodel")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the PyTorch checkpoint (.pth)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the CoreML .mlmodel file")
    parser.add_argument("--num_classes", type=int, default=14,
                        help="Number of output classes")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size (height=width)")
    args = parser.parse_args()

    model = load_model(args.checkpoint, num_classes=args.num_classes)
    convert_to_coreml(model, args.output, image_size=args.image_size)


if __name__ == "__main__":
    main()
