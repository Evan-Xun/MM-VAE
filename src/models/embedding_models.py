from torch import nn


class ImageEmbeddingBaseline(nn.Module):
    def __init__(self, image_embedding_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(image_embedding_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, image_embedding):
        return self.classifier(image_embedding).squeeze(-1)


class LateFusionEmbedding(nn.Module):
    def __init__(self, cbc_dim, image_embedding_dim=512, use_missing_mask=False):
        super().__init__()
        input_dim = cbc_dim * 2 if use_missing_mask else cbc_dim
        self.use_missing_mask = use_missing_mask
        self.cbc_encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(32 + image_embedding_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, cbc, image_embedding, cbc_mask=None):
        import torch
        if self.use_missing_mask:
            if cbc_mask is None:
                raise ValueError("cbc_mask is required when use_missing_mask=True")
            cbc = torch.cat([cbc, cbc_mask], dim=1)
        return self.classifier(torch.cat([self.cbc_encoder(cbc), image_embedding], dim=1)).squeeze(-1)
