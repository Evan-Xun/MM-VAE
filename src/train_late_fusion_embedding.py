import argparse

from models.embedding_models import LateFusionEmbedding
from train_utils import collect_predictions, device, fit_classifier, make_loader, prepare_cbc, save_outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-set", default="core_without_hgb")
    p.add_argument("--embeddings", default="data/processed/image_embeddings/resnet18_mean.csv")
    p.add_argument("--pooling", default="mean")
    p.add_argument("--experiment-name", default="G4_late_fusion_without_hgb")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable-missing-mask", action="store_true")
    args = p.parse_args()
    use_missing_mask = not args.disable_missing_mask
    scaler, impute_values, cbc_dim, _ = prepare_cbc(args.feature_set)
    train = make_loader("train", args.feature_set, args.batch_size, True, scaler, impute_values, cbc=True, load_images=False, image_embeddings_csv=args.embeddings)
    val = make_loader("val", args.feature_set, args.batch_size, False, scaler, impute_values, cbc=True, load_images=False, image_embeddings_csv=args.embeddings)
    test = make_loader("test", args.feature_set, args.batch_size, False, scaler, impute_values, cbc=True, load_images=False, image_embeddings_csv=args.embeddings)
    dim = train.dataset.image_embedding_dim
    model = fit_classifier(
        LateFusionEmbedding(cbc_dim, dim, use_missing_mask=use_missing_mask),
        train,
        val,
        args,
        "late_embedding",
    )
    case_ids, y, prob = collect_predictions(model, test, device(), "late_embedding")
    print(save_outputs(args.experiment_name, "Late Fusion", args.feature_set, "12 patches", args.pooling, "", case_ids, y, prob))


if __name__ == "__main__":
    main()
