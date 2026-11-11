from torch import nn


class CBCBaseline(nn.Module):
    def __init__(self, cbc_dim, use_missing_mask=False):
        super().__init__()
        input_dim = cbc_dim * 2 if use_missing_mask else cbc_dim
        self.use_missing_mask = use_missing_mask
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, cbc, cbc_mask=None):
        if self.use_missing_mask:
            if cbc_mask is None:
                raise ValueError("cbc_mask is required when use_missing_mask=True")
            cbc = __import__("torch").cat([cbc, cbc_mask], dim=1)
        return self.net(cbc).squeeze(-1)
