import argparse

from models.embedding_models import ImageEmbeddingBaseline
from train_utils import collect_predictions, device, fit_classifier, make_loader, save_outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", default="data/processed/image_embeddings/resnet18_mean.csv")
    p.add_argument("--pooling", default="mean")
    p.add_argument("--experiment-name", default="G3_image_only_resnet18_mean")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train = make_loader("train", "core_without_hgb", args.batch_size, True, cbc=False, load_images=False, image_embeddings_csv=args.embeddings)
    val = make_loader("val", "core_without_hgb", args.batch_size, False, cbc=False, load_images=False, image_embeddings_csv=args.embeddings)
    test = make_loader("test", "core_without_hgb", args.batch_size, False, cbc=False, load_images=False, image_embeddings_csv=args.embeddings)
    dim = train.dataset.image_embedding_dim
    model = fit_classifier(ImageEmbeddingBaseline(dim), train, val, args, "image_embedding")
    case_ids, y, prob = collect_predictions(model, test, device(), "image_embedding")
    print(save_outputs(args.experiment_name, "Image-only", "No", "12 patches", args.pooling, "", case_ids, y, prob))


if __name__ == "__main__":
    main()
