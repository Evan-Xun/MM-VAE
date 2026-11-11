import argparse

from models.cbc_baseline import CBCBaseline
from train_utils import collect_predictions, fit_classifier, make_loader, prepare_cbc, save_outputs, device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-set", default="core_without_hgb")
    p.add_argument("--experiment-name", default="G2_cbc_core_without_hgb")
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
    train = make_loader("train", args.feature_set, args.batch_size, True, scaler, impute_values, cbc=True, load_images=False)
    val = make_loader("val", args.feature_set, args.batch_size, False, scaler, impute_values, cbc=True, load_images=False)
    test = make_loader("test", args.feature_set, args.batch_size, False, scaler, impute_values, cbc=True, load_images=False)
    model = fit_classifier(CBCBaseline(cbc_dim, use_missing_mask=use_missing_mask), train, val, args, "cbc")
    case_ids, y, prob = collect_predictions(model, test, device(), "cbc")
    print(save_outputs(args.experiment_name, "CBC-only", args.feature_set, "No", "", "", case_ids, y, prob))


if __name__ == "__main__":
    main()
