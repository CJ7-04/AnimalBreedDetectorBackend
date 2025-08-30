import torch
from torch import nn
from torchvision import models
import timm

def build_model(num_classes: int, arch: str = "efficientnet_b0", pretrained: bool = True):
    arch = arch.lower()
    if arch.startswith("efficientnet_b"):
        fn = getattr(models, arch)
        weights = models.get_model_weights(arch).DEFAULT if pretrained else None
        m = fn(weights=weights)
        # torchvision EfficientNet: classifier is (Dropout, Linear) commonly
        try:
            in_feats = m.classifier[1].in_features
            m.classifier[1] = nn.Linear(in_feats, num_classes)
        except Exception:
            # fallback replace head for timm models
            m.classifier = nn.Linear(getattr(m, 'classifier').in_features, num_classes)
        return m
    elif arch in {"resnet18", "resnet34", "resnet50"}:
        fn = getattr(models, arch)
        weights = models.get_model_weights(arch).DEFAULT if pretrained else None
        m = fn(weights=weights)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
        return m
    else:
        m = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
        return m
