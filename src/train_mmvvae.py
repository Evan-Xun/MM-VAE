import argparse

import torch
from torch import nn

from config import RESULTS_DIR
from models.mmvvae import MMVAE
from train_utils import collect_predictions, device, make_loader, prepare_cbc, save_outputs
from utils import binary_metrics, read_csv, set_seed, write_csv


def kl_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def masked_mse(recon, target, observed_mask):
    diff = (recon - target).pow(2)
    if observed_mask is None:
        return diff.mean()
    weighted = diff * observed_mask
    return weighted.sum() / observed_mask.sum().clamp_min(1.0)


def train(args):
    set_seed(args.seed)
    use_missing_mask = not args.disable_missing_mask
    scaler, impute_values, cbc_dim, _ = prepare_cbc(args.feature_set)
    train_loader = make_loader("train", args.feature_set, args.batch_size, True, scaler, impute_values, cbc=True)
    val_loader = make_loader("val", args.feature_set, args.batch_size, False, scaler, impute_values, cbc=True)
    test_loader = make_loader("test", args.feature_set, args.batch_size, False, scaler, impute_values, cbc=True)
    dev = device()
    model = MMVAE(
        cbc_dim, args.latent_dim, args.backbone, args.pooling, args.pretrained, use_missing_mask=use_missing_mask,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    best_val = -1
    patience = 0
    history = []
    ckpt = RESULTS_DIR / "checkpoints" / f"{args.experiment_name}_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"total_loss": 0.0, "cbc_recon_loss": 0.0, "img_recon_loss": 0.0, "kl_loss": 0.0, "cls_loss": 0.0}
        beta = min(1.0, epoch / max(1, args.kl_warmup_epochs))
        for batch in train_loader:
            cbc = batch["cbc"].to(dev)
            cbc_mask = batch["cbc_mask"].to(dev)
            patches = batch["patches"].to(dev)
            labels = batch["label"].to(dev)
            opt.zero_grad()
            out = model(cbc, patches, cbc_mask=cbc_mask, mode="complete")
            cbc_recon = masked_mse(out["cbc_recon"], cbc, cbc_mask)
            img_recon = mse(out["img_recon"], out["image_embedding"].detach())
            kl = kl_loss(out["mu_z"], out["logvar_z"])
            cls = bce(out["logits"], labels)
            loss = cbc_recon + args.alpha * img_recon + beta * args.beta * kl + args.lambda_cls * cls
            loss.backward()
            opt.step()
            bs = labels.numel()
            for key, value in [
                ("total_loss", loss), ("cbc_recon_loss", cbc_recon), ("img_recon_loss", img_recon),
                ("kl_loss", kl), ("cls_loss", cls),
            ]:
                totals[key] += float(value.item()) * bs
        _, y_val, p_val = collect_predictions(model, val_loader, dev, "mmvvae")
        val_metrics = binary_metrics(y_val, p_val)
        row = {"epoch": epoch, "beta": beta, **{k: v / len(train_loader.dataset) for k, v in totals.items()}, **val_metrics}
        history.append(row)
        score = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else val_metrics["f1"]
        if score > best_val:
            best_val = score
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "cbc_dim": cbc_dim,
                "feature_set": args.feature_set,
                "pooling": args.pooling,
                "latent_dim": args.latent_dim,
                "use_missing_mask": use_missing_mask,
            }, ckpt)
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                break
    write_csv(RESULTS_DIR / "logs" / f"{args.experiment_name}_history.csv", history, list(history[0].keys()))
    model.load_state_dict(torch.load(ckpt, map_location=dev)["model_state"])
    case_ids, y, prob = collect_predictions(model, test_loader, dev, "mmvvae")
    metrics = save_outputs(
        args.experiment_name, "MM-VAE", args.feature_set, "12 patches", args.pooling,
        args.latent_dim, case_ids, y, prob,
    )
    print(metrics)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-set", default="core_without_hgb")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--pooling", default="mean", choices=["mean", "attention"])
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--kl-warmup-epochs", type=int, default=20)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lambda-cls", type=float, default=1.0)
    p.add_argument("--experiment-name", default="G5_mmvvae_without_hgb")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable-missing-mask", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
