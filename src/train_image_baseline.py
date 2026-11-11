import argparse

from models.image_baseline import ImageBaseline
from train_utils import collect_predictions, fit_classifier, make_loader, save_outputs, device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--pooling", default="mean", choices=["mean", "attention"])
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--experiment-name", default="G3_image_only_resnet18_mean")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train = make_loader("train", "core_without_hgb", args.batch_size, True, cbc=False)
    val = make_loader("val", "core_without_hgb", args.batch_size, False, cbc=False)
    test = make_loader("test", "core_without_hgb", args.batch_size, False, cbc=False)
    model = fit_classifier(ImageBaseline(args.backbone, args.pooling, args.pretrained), train, val, args, "image")
    case_ids, y, prob = collect_predictions(model, test, device(), "image")
    print(save_outputs(args.experiment_name, "Image-only", "No", "12 patches", args.pooling, "", case_ids, y, prob))


if __name__ == "__main__":
    main()
