import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x):
        weights = torch.softmax(self.net(x), dim=1)
        return (weights * x).sum(dim=1)


class SmallCNN(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return self.proj(self.features(x).flatten(1))


def build_backbone(name="resnet18", pretrained=False, freeze=True):
    if name == "smallcnn":
        return SmallCNN(), 512
    try:
        from torchvision import models
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity()
    except Exception:
        model, dim = SmallCNN(), 512
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    return model, dim


class PatchImageEncoder(nn.Module):
    def __init__(self, backbone="resnet18", pooling="mean", pretrained=False, freeze=True):
        super().__init__()
        self.backbone, self.embedding_dim = build_backbone(backbone, pretrained, freeze)
        self.pooling = pooling
        self.attn = AttentionPooling(self.embedding_dim) if pooling == "attention" else None

    def forward(self, patches):
        b, n, c, h, w = patches.shape
        patch_emb = self.backbone(patches.view(b * n, c, h, w)).view(b, n, -1)
        if self.attn is not None:
            return self.attn(patch_emb)
        return patch_emb.mean(dim=1)
