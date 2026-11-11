from pathlib import Path
import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import FEATURE_SETS, PROCESSED_DIR, RESULTS_DIR
from datasets import PatientLevelDataset, Standardizer, impute_by_train, load_cbc_table
from utils import append_summary, binary_metrics, read_csv, set_seed, write_csv, write_json


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_cbc(feature_set, manifest_dir=PROCESSED_DIR, cbc_csv=None):
    cbc_csv = cbc_csv or Path(manifest_dir) / "cbc_features.csv"
    train_rows = read_csv(Path(manifest_dir) / "train.csv")
    cbc_by_case, _, names = load_cbc_table(cbc_csv, feature_set)
    cbc_by_case, impute_values = impute_by_train(cbc_by_case, [r["case_id"] for r in train_rows])
    scaler = Standardizer().fit([cbc_by_case[r["case_id"]] for r in train_rows])
    scaler_path = RESULTS_DIR / "checkpoints" / "cbc_scaler.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump({"feature_set": feature_set, "mean": scaler.mean, "std": scaler.std, "impute_values": impute_values}, f)
    return scaler, impute_values, len(FEATURE_SETS[feature_set]), names


def make_loader(split, feature_set, batch_size, train, scaler=None, impute_values=None,
                manifest_dir=PROCESSED_DIR, cbc=True, image_size=224, load_images=True,
                image_embeddings_csv=None):
    ds = PatientLevelDataset(
        Path(manifest_dir) / f"{split}.csv",
        Path(manifest_dir) / "cbc_features.csv" if cbc else None,
        feature_set=feature_set,
        train=train,
        scaler=scaler,
        impute_values=impute_values,
        image_size=image_size,
        load_images=load_images,
        image_embeddings_csv=image_embeddings_csv,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=0)


def collect_predictions(model, loader, dev, kind, mode="complete"):
    model.eval()
    probs, labels, case_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels.extend(batch["label"].cpu().numpy().astype(int).tolist())
            case_ids.extend(batch["case_id"])
            cbc_mask = batch.get("cbc_mask")
            if kind == "cbc":
                logits = model(batch["cbc"].to(dev), cbc_mask.to(dev) if cbc_mask is not None else None)
            elif kind == "image":
                logits = model(batch["patches"].to(dev))
            elif kind == "image_embedding":
                logits = model(batch["image_embedding"].to(dev))
            elif kind == "late":
                logits = model(batch["cbc"].to(dev), batch["patches"].to(dev), cbc_mask.to(dev) if cbc_mask is not None else None)
            elif kind == "late_embedding":
                logits = model(
                    batch["cbc"].to(dev),
                    batch["image_embedding"].to(dev),
                    cbc_mask.to(dev) if cbc_mask is not None else None,
                )
            else:
                cbc = batch.get("cbc")
                if "image_embedding" in batch:
                    image_embedding = batch.get("image_embedding")
                    out = model(
                        cbc.to(dev) if cbc is not None and mode != "missing_cbc" else None,
                        image_embedding.to(dev) if image_embedding is not None and mode != "missing_image" else None,
                        cbc_mask.to(dev) if cbc_mask is not None and mode != "missing_cbc" else None,
                        mode=mode,
                    )
                else:
                    patches = batch.get("patches")
                    out = model(
                        cbc.to(dev) if cbc is not None and mode != "missing_cbc" else None,
                        patches.to(dev) if patches is not None and mode != "missing_image" else None,
                        cbc_mask.to(dev) if cbc_mask is not None and mode != "missing_cbc" else None,
                        mode=mode,
                    )
                logits = out["logits"]
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return case_ids, labels, probs


def save_outputs(experiment_name, model_label, feature_set, image_input, pooling, latent_dim,
                 test_case_ids, y_true, y_prob):
    metrics = binary_metrics(y_true, y_prob)
    metrics_dir = RESULTS_DIR / "metrics"
    preds_dir = RESULTS_DIR / "predictions"
    write_json(metrics_dir / f"{experiment_name}_metrics.json", metrics)
    pred_rows = []
    for cid, y, p in zip(test_case_ids, y_true, y_prob):
        pred_rows.append({
            "case_id": cid,
            "true_label": y,
            "predicted_probability": p,
            "predicted_label": int(p >= 0.5),
            "split": "test",
            "experiment_name": experiment_name,
        })
    write_csv(
        preds_dir / f"{experiment_name}_test_predictions.csv",
        pred_rows,
        ["case_id", "true_label", "predicted_probability", "predicted_label", "split", "experiment_name"],
    )
    append_summary(metrics_dir / "all_results_summary.csv", {
        "experiment_name": experiment_name,
        "model": model_label,
        "feature_set": feature_set,
        "image_input": image_input,
        "pooling": pooling,
        "latent_dim": latent_dim,
        **metrics,
    })
    return metrics


def fit_classifier(model, train_loader, val_loader, args, kind):
    set_seed(args.seed)
    dev = device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_val = -1
    patience = 0
    history = []
    ckpt = RESULTS_DIR / "checkpoints" / f"{args.experiment_name}_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            opt.zero_grad()
            labels = batch["label"].to(dev)
            cbc_mask = batch.get("cbc_mask")
            if kind == "cbc":
                logits = model(batch["cbc"].to(dev), cbc_mask.to(dev) if cbc_mask is not None else None)
            elif kind == "image":
                logits = model(batch["patches"].to(dev))
            elif kind == "image_embedding":
                logits = model(batch["image_embedding"].to(dev))
            elif kind == "late_embedding":
                logits = model(
                    batch["cbc"].to(dev),
                    batch["image_embedding"].to(dev),
                    cbc_mask.to(dev) if cbc_mask is not None else None,
                )
            else:
                logits = model(
                    batch["cbc"].to(dev),
                    batch["patches"].to(dev),
                    cbc_mask.to(dev) if cbc_mask is not None else None,
                )
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            total += float(loss.item()) * labels.numel()
        _, y_val, p_val = collect_predictions(model, val_loader, dev, kind)
        val_metrics = binary_metrics(y_val, p_val)
        history.append({"epoch": epoch, "train_loss": total / len(train_loader.dataset), **val_metrics})
        score = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else val_metrics["f1"]
        if score > best_val:
            best_val = score
            patience = 0
            payload = {"model_state": model.state_dict(), "args": vars(args)}
            if hasattr(args, "disable_missing_mask"):
                payload["use_missing_mask"] = not args.disable_missing_mask
            torch.save(payload, ckpt)
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                break
    write_csv(RESULTS_DIR / "logs" / f"{args.experiment_name}_history.csv", history, list(history[0].keys()))
    model.load_state_dict(torch.load(ckpt, map_location=dev)["model_state"])
    return model
