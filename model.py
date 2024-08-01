import torchvision.models as models
from torchvision.models.resnet import ResNet34_Weights
import torch.nn as nn
import torch

def build_model(pretrained=True, fine_tune=True, num_classes=1):
    if pretrained:
        print("[INFO]: Loading pretrained weights")
    elif not pretrained:
        print("[INFO]: Not Loading pretrained weights")

    # model = models.resnet34(pretrained=pretrained)
    model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    if fine_tune:
        print("[INFO]: Fine-tuning all layers...")
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print("[INFO]: Freezing hidden layers...")
        for params in model.parameters():
            params.requires_grad = False

    model.fc = nn.Linear(512, num_classes)

    return model