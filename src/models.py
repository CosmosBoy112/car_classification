"""Backbone loading and head replacement."""
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

import config as C

# Same weights for the frozen baseline and for fine-tuning, so the 2x2 table
# compares training strategy, not starting point.
WEIGHTS = ResNet50_Weights.IMAGENET1K_V2


def build_classifier(num_classes: int = C.NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ResNet50 with the 1000-class ImageNet head replaced by a fresh 196-class head."""
    model = resnet50(weights=WEIGHTS if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_feature_extractor() -> nn.Module:
    """Frozen ImageNet ResNet50 that outputs the 2048-d pooled feature."""
    model = resnet50(weights=WEIGHTS)
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    return model.eval()


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float):
    """Two learning rates: small for pretrained layers, larger for the new head.

    Biases and BatchNorm weights get no weight decay (decaying them only hurts).
    """
    groups = {k: {"params": [], "lr": lr, "weight_decay": wd}
              for k, lr, wd in [("bb", lr_backbone, weight_decay), ("bb_nd", lr_backbone, 0.0),
                                ("head", lr_head, weight_decay), ("head_nd", lr_head, 0.0)]}
    for name, p in model.named_parameters():
        key = "head" if name.startswith("fc.") else "bb"
        if p.ndim <= 1:
            key += "_nd"
        groups[key]["params"].append(p)
    return [g for g in groups.values() if g["params"]]


def load_checkpoint(path, device) -> nn.Module:
    model = build_classifier(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()
