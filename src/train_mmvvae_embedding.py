import argparse

import torch
from torch import nn

from config import RESULTS_DIR
from models.mmvvae import MMVAEFromEmbedding
from train_mmvvae import kl_loss, masked_mse
from train_utils import collect_predictions, device, make_loader, prepare_cbc, save_outputs
from utils import binary_metrics, set_seed, write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-set", default="core_without_hgb")
    p.add_argument("--embeddings", default="data/processed/image_embeddings/resnet18_mean.csv")
    p.add_argument("--pooling", default="mean")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--kl-warmup-epochs", type=int, default=20)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lambda-cls", type=float, default=1.0)
    p.add_argument("--experiment-name", default="G5_mmvvae_without_hgb")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable-missing-mask", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    use_missing_mask = not args.disable_missing_mask
    scaler, impute_values, cbc_dim, _ = prepare_cbc(args.feature_set)
    train_loader = make_loader("train", args.feature_set, args.batch_size, True, scaler, impute_values, cbc=True, load_images=False, image_embeddings_csv=args.embeddings)
    val_loader = make_loader("val", args.feature_set, args.batch_size, False, scaler, impute_values, cbc=True, load_images=False, image_embeddings_csv=args.embeddings)
    test_loader = make_loader("test", args.feature_set, args.batch_size, False, scaler, impute_values, cbc=True, load_images=False, image_embeddings_csv=args.embeddings)
    dim = train_loader.dataset.image_embedding_dim
    dev = device()
    model = MMVAEFromEmbedding(cbc_dim, dim, args.latent_dim, use_missing_mask=use_missing_mask).to(dev)
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
        beta = min(1.0, epoch / max(1, args.kl_warmup_epochs))
        totals = {"total_loss": 0.0, "cbc_recon_loss": 0.0, "img_recon_loss": 0.0, "kl_loss": 0.0, "cls_loss": 0.0}
        for batch in train_loader:
            cbc = batch["cbc"].to(dev)
            cbc_mask = batch["cbc_mask"].to(dev)
            image_embedding = batch["image_embedding"].to(dev)
            labels = batch["label"].to(dev)
            opt.zero_grad()
            out = model(cbc, image_embedding, cbc_mask=cbc_mask, mode="complete")
            cbc_recon = masked_mse(out["cbc_recon"], cbc, cbc_mask)
            img_recon = mse(out["img_recon"], image_embedding)
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
        history.append({"epoch": epoch, "beta": beta, **{k: v / len(train_loader.dataset) for k, v in totals.items()}, **val_metrics})
        score = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else val_metrics["f1"]
        if score > best_val:
            best_val = score
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "feature_set": args.feature_set,
                "pooling": args.pooling,
                "latent_dim": args.latent_dim,
                "image_embedding_dim": dim,
                "cbc_dim": cbc_dim,
                "use_missing_mask": use_missing_mask,
                "args": vars(args),
            }, ckpt)
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                break

    write_csv(RESULTS_DIR / "logs" / f"{args.experiment_name}_history.csv", history, list(history[0].keys()))
    model.load_state_dict(torch.load(ckpt, map_location=dev)["model_state"])
    case_ids, y, prob = collect_predictions(model, test_loader, dev, "mmvvae")
    print(save_outputs(args.experiment_name, "MM-VAE", args.feature_set, "12 patches", args.pooling, args.latent_dim, case_ids, y, prob))


if __name__ == "__main__":
    main()
