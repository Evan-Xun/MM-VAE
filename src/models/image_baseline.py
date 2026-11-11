from torch import nn

from image_encoder import PatchImageEncoder


class ImageBaseline(nn.Module):
    def __init__(self, backbone="resnet18", pooling="mean", pretrained=False, freeze=True):
        super().__init__()
        self.image_encoder = PatchImageEncoder(backbone, pooling, pretrained, freeze)
        self.classifier = nn.Sequential(
            nn.Linear(self.image_encoder.embedding_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, patches):
        emb = self.image_encoder(patches)
        return self.classifier(emb).squeeze(-1)
