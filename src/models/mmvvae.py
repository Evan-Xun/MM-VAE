import torch
from torch import nn

from image_encoder import PatchImageEncoder


class GaussianEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        last = in_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(last, dim), nn.ReLU()])
            last = dim
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(last, latent_dim)
        self.logvar = nn.Linear(last, latent_dim)

    def forward(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h)


class MMVAE(nn.Module):
    def __init__(self, cbc_dim, latent_dim=16, backbone="resnet18", pooling="mean", pretrained=False, freeze=True,
                 use_missing_mask=False):
        super().__init__()
        self.image_backbone = PatchImageEncoder(backbone, pooling, pretrained, freeze)
        self.image_embedding_dim = self.image_backbone.embedding_dim
        cbc_input_dim = cbc_dim * 2 if use_missing_mask else cbc_dim
        self.use_missing_mask = use_missing_mask
        self.cbc_encoder = GaussianEncoder(cbc_input_dim, [64, 64], latent_dim)
        self.img_encoder = GaussianEncoder(self.image_embedding_dim, [256, 128], latent_dim)
        self.cbc_decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, cbc_dim))
        self.img_decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, self.image_embedding_dim))
        self.classifier = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.latent_dim = latent_dim

    def _cbc_input(self, cbc, cbc_mask):
        if not self.use_missing_mask:
            return cbc
        if cbc_mask is None:
            raise ValueError("cbc_mask is required when use_missing_mask=True")
        return torch.cat([cbc, cbc_mask], dim=1)

    def product_of_experts(self, experts, device):
        mu = [torch.zeros(experts[0][0].shape, device=device)]
        logvar = [torch.zeros(experts[0][1].shape, device=device)]
        for m, lv in experts:
            mu.append(m)
            logvar.append(lv)
        mu = torch.stack(mu)
        logvar = torch.stack(logvar)
        var = torch.exp(logvar).clamp_min(1e-8)
        precision = 1.0 / var
        precision_sum = precision.sum(dim=0)
        fused_mu = (mu * precision).sum(dim=0) / precision_sum
        fused_var = 1.0 / precision_sum
        return fused_mu, torch.log(fused_var.clamp_min(1e-8))

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, cbc=None, patches=None, cbc_mask=None, mode="complete"):
        experts = []
        image_embedding = None
        if mode in ("complete", "missing_image") and cbc is not None:
            experts.append(self.cbc_encoder(self._cbc_input(cbc, cbc_mask)))
        if mode in ("complete", "missing_cbc") and patches is not None:
            image_embedding = self.image_backbone(patches)
            experts.append(self.img_encoder(image_embedding))
        if not experts:
            raise ValueError("At least one modality must be present")
        mu_z, logvar_z = self.product_of_experts(experts, experts[0][0].device)
        z = self.reparameterize(mu_z, logvar_z)
        return {
            "logits": self.classifier(z).squeeze(-1),
            "z": z,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "cbc_recon": self.cbc_decoder(z),
            "img_recon": self.img_decoder(z),
            "image_embedding": image_embedding,
        }


class MMVAEFromEmbedding(nn.Module):
    def __init__(self, cbc_dim, image_embedding_dim=512, latent_dim=16, use_missing_mask=False):
        super().__init__()
        self.image_embedding_dim = image_embedding_dim
        cbc_input_dim = cbc_dim * 2 if use_missing_mask else cbc_dim
        self.use_missing_mask = use_missing_mask
        self.cbc_encoder = GaussianEncoder(cbc_input_dim, [64, 64], latent_dim)
        self.img_encoder = GaussianEncoder(image_embedding_dim, [256, 128], latent_dim)
        self.cbc_decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, cbc_dim))
        self.img_decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, image_embedding_dim))
        self.classifier = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.latent_dim = latent_dim

    def _cbc_input(self, cbc, cbc_mask):
        if not self.use_missing_mask:
            return cbc
        if cbc_mask is None:
            raise ValueError("cbc_mask is required when use_missing_mask=True")
        return torch.cat([cbc, cbc_mask], dim=1)

    def product_of_experts(self, experts, device):
        mu = [torch.zeros(experts[0][0].shape, device=device)]
        logvar = [torch.zeros(experts[0][1].shape, device=device)]
        for m, lv in experts:
            mu.append(m)
            logvar.append(lv)
        mu = torch.stack(mu)
        logvar = torch.stack(logvar)
        var = torch.exp(logvar).clamp_min(1e-8)
        precision = 1.0 / var
        precision_sum = precision.sum(dim=0)
        fused_mu = (mu * precision).sum(dim=0) / precision_sum
        fused_var = 1.0 / precision_sum
        return fused_mu, torch.log(fused_var.clamp_min(1e-8))

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, cbc=None, image_embedding=None, cbc_mask=None, mode="complete"):
        experts = []
        if mode in ("complete", "missing_image") and cbc is not None:
            experts.append(self.cbc_encoder(self._cbc_input(cbc, cbc_mask)))
        if mode in ("complete", "missing_cbc") and image_embedding is not None:
            experts.append(self.img_encoder(image_embedding))
        if not experts:
            raise ValueError("At least one modality must be present")
        mu_z, logvar_z = self.product_of_experts(experts, experts[0][0].device)
        z = self.reparameterize(mu_z, logvar_z)
        return {
            "logits": self.classifier(z).squeeze(-1),
            "z": z,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "cbc_recon": self.cbc_decoder(z),
            "img_recon": self.img_decoder(z),
            "image_embedding": image_embedding,
        }
