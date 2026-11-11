from torch import nn

from image_encoder import PatchImageEncoder


class LateFusion(nn.Module):
    def __init__(self, cbc_dim, backbone="resnet18", pooling="mean", pretrained=False, freeze=True, use_missing_mask=False):
        super().__init__()
        input_dim = cbc_dim * 2 if use_missing_mask else cbc_dim
        self.use_missing_mask = use_missing_mask
        self.cbc_encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.image_encoder = PatchImageEncoder(backbone, pooling, pretrained, freeze)
        self.classifier = nn.Sequential(
            nn.Linear(32 + self.image_encoder.embedding_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, cbc, patches, cbc_mask=None):
        if self.use_missing_mask:
            if cbc_mask is None:
                raise ValueError("cbc_mask is required when use_missing_mask=True")
            cbc = __import__("torch").cat([cbc, cbc_mask], dim=1)
        return self.classifier(
            __import__("torch").cat([self.cbc_encoder(cbc), self.image_encoder(patches)], dim=1)
        ).squeeze(-1)
