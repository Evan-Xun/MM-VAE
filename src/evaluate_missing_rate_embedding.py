import argparse
import statistics

import torch

from config import RESULTS_DIR
from models.mmvvae import MMVAEFromEmbedding
from train_utils import device, make_loader, prepare_cbc
from utils import binary_metrics, set_seed, write_csv


def parse_rates(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def evaluate_rate(model, loader, dev, scenario, missing_rate, seed):
    set_seed(seed)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    model.eval()
    probs, labels = [], []

    with torch.no_grad():
        for batch in loader:
            cbc = batch["cbc"].to(dev)
            image_embedding = batch["image_embedding"].to(dev)
            cbc_mask = batch["cbc_mask"].to(dev)
            batch_labels = batch["label"].cpu().numpy().astype(int).tolist()

            complete_logits = model(cbc, image_embedding, cbc_mask, mode="complete")["logits"]
            if scenario == "image_missing":
                missing_logits = model(cbc, None, cbc_mask, mode="missing_image")["logits"]
            elif scenario == "cbc_missing":
                missing_logits = model(None, image_embedding, None, mode="missing_cbc")["logits"]
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            missing_selector = torch.rand(len(batch_labels), generator=gen) < missing_rate
            missing_selector = missing_selector.to(dev)
            logits = torch.where(missing_selector, missing_logits, complete_logits)

            labels.extend(batch_labels)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    return binary_metrics(labels, probs)


def summarize(rows, rates, scenarios, metric_names):
    summary = []
    for scenario in scenarios:
        for rate in rates:
            subset = [
                row for row in rows
                if row["scenario"] == scenario and abs(float(row["missing_rate"]) - rate) < 1e-12
            ]
            item = {"scenario": scenario, "missing_rate": rate, "n_repeats": len(subset)}
            for metric in metric_names:
                vals = [float(row[metric]) for row in subset if row[metric] != ""]
                item[f"{metric}_mean"] = statistics.mean(vals) if vals else ""
                item[f"{metric}_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            summary.append(item)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/checkpoints/G7_mmvvae_with_hgb_best.pt")
    p.add_argument("--embeddings", default="data/processed/image_embeddings/resnet18_mean.csv")
    p.add_argument("--rates", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-prefix", default="missing_rate_simulation_with_hgb")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    feature_set = ckpt.get("feature_set", "core_with_hgb")
    latent_dim = int(ckpt.get("latent_dim", 16))
    use_missing_mask = bool(ckpt.get("use_missing_mask", not ckpt_args.get("disable_missing_mask", True)))

    scaler, impute_values, cbc_dim, _ = prepare_cbc(feature_set)
    loader = make_loader(
        "test",
        feature_set,
        args.batch_size,
        False,
        scaler,
        impute_values,
        cbc=True,
        load_images=False,
        image_embeddings_csv=args.embeddings,
    )
    image_dim = int(ckpt.get("image_embedding_dim", loader.dataset.image_embedding_dim))
    model = MMVAEFromEmbedding(cbc_dim, image_dim, latent_dim, use_missing_mask=use_missing_mask)
    model.load_state_dict(ckpt["model_state"])
    dev = device()
    model.to(dev)

    rates = parse_rates(args.rates)
    seeds = [int(s) for s in parse_rates(args.seeds)]
    scenarios = ["image_missing", "cbc_missing"]
    metric_names = ["accuracy", "precision", "recall", "specificity", "npv", "f1", "roc_auc"]
    count_names = ["tn", "fp", "fn", "tp"]

    rows = []
    for scenario in scenarios:
        for rate in rates:
            for seed in seeds:
                metrics = evaluate_rate(model, loader, dev, scenario, rate, seed)
                rows.append({
                    "scenario": scenario,
                    "missing_rate": rate,
                    "seed": seed,
                    **{name: metrics[name] for name in metric_names + count_names},
                })

    detail_fields = ["scenario", "missing_rate", "seed", *metric_names, *count_names]
    detail_path = RESULTS_DIR / "metrics" / f"{args.output_prefix}.csv"
    write_csv(detail_path, rows, detail_fields)

    summary_rows = summarize(rows, rates, scenarios, metric_names)
    summary_fields = ["scenario", "missing_rate", "n_repeats"]
    for metric in metric_names:
        summary_fields.extend([f"{metric}_mean", f"{metric}_sd"])
    summary_path = RESULTS_DIR / "metrics" / f"{args.output_prefix}_summary.csv"
    write_csv(summary_path, summary_rows, summary_fields)

    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
