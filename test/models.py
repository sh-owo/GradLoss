import torch.nn as nn
import torchvision.models as tvm


_MODELS = {
    "vgg16": (tvm.vgg16, tvm.VGG16_Weights.DEFAULT),
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.DEFAULT),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.DEFAULT),
}


def create_model(arch, num_classes, pretrained=True):
    if arch not in _MODELS:
        raise ValueError(f"unknown arch: {arch}")
    builder, weights = _MODELS[arch]
    model = builder(weights=weights if pretrained else None)

    if arch == "vgg16":
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    else:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model


def get_target_layer(model, arch):
    if arch == "vgg16":
        for m in reversed(model.features):
            if isinstance(m, nn.Conv2d):
                return m
        raise RuntimeError("VGG16: conv layer not found")
    if arch in ("resnet18", "resnet50"):
        return model.layer4[-1]
    raise ValueError(f"unknown arch: {arch}")
