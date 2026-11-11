import argparse
import statistics

from config import RESULTS_DIR
from utils import append_summary, binary_metrics, read_csv, write_csv, write_json


def train_median(cbc_rows, split_rows, feature):
    cbc_by_case = {row["case_id"]: row for row in cbc_rows}
    values = []
    for row in split_rows:
        value = cbc_by_case[row["case_id"]].get(feature, "")
        if value != "":
            values.append(float(value))
    if not values:
        raise ValueError(f"No non-missing training values for {feature}")
    return statistics.median(values)


def score_hgb_rule(split_rows, cbc_rows, threshold, missing_value, experiment_name):
    cbc_by_case = {row["case_id"]: row for row in cbc_rows}
    y_true = []
    y_prob = []
    pred_rows = []
    missing_count = 0
    for row in split_rows:
        case_id = row["case_id"]
        value = cbc_by_case[case_id].get("HGB", "")
        if value == "":
            missing_count += 1
        hgb = float(value) if value != "" else missing_value
        pred = 1 if hgb < threshold else 0
        # Rule output is deterministic, so probability is encoded as 0/1.
        prob = float(pred)
        label = int(row["label_id"])
        y_true.append(label)
        y_prob.append(prob)
        pred_rows.append({
            "case_id": case_id,
            "true_label": label,
            "hgb": hgb,
            "threshold": threshold,
            "predicted_probability": prob,
            "predicted_label": pred,
            "split": row["split"],
            "experiment_name": experiment_name,
        })
    return binary_metrics(y_true, y_prob), pred_rows, missing_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=13.0)
    parser.add_argument("--experiment-name", default="G0_rule_hgb_lt_13")
    parser.add_argument("--manifest-dir", default="data/processed")
    parser.add_argument("--cbc", default="data/processed/cbc_features.csv")
    args = parser.parse_args()

    cbc_rows = read_csv(args.cbc)
    train_rows = read_csv(f"{args.manifest_dir}/train.csv")
    test_rows = read_csv(f"{args.manifest_dir}/test.csv")
    missing_hgb = train_median(cbc_rows, train_rows, "HGB")
    metrics, pred_rows, missing_count = score_hgb_rule(
        test_rows, cbc_rows, args.threshold, missing_hgb, args.experiment_name,
    )
    metrics["threshold"] = args.threshold
    metrics["missing_hgb_imputation"] = missing_hgb
    metrics["test_rows_after_hgb_imputation"] = len(test_rows)
    metrics["imputed_missing_hgb"] = missing_count

    write_json(RESULTS_DIR / "metrics" / f"{args.experiment_name}_metrics.json", metrics)
    write_csv(
        RESULTS_DIR / "predictions" / f"{args.experiment_name}_test_predictions.csv",
        pred_rows,
        [
            "case_id", "true_label", "hgb", "threshold", "predicted_probability",
            "predicted_label", "split", "experiment_name",
        ],
    )
    append_summary(RESULTS_DIR / "metrics" / "all_results_summary.csv", {
        "experiment_name": args.experiment_name,
        "model": "Rule-based HGB threshold",
        "feature_set": f"HGB < {args.threshold:g}",
        "image_input": "No",
        "pooling": "",
        "latent_dim": "",
        **{k: v for k, v in metrics.items() if k not in {"threshold", "missing_hgb_imputation"}},
    })
    print(metrics)


if __name__ == "__main__":
    main()
