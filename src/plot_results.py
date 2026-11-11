import argparse
from pathlib import Path

from config import RESULTS_DIR
from utils import read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

    pred_path = RESULTS_DIR / "predictions" / f"{args.experiment_name}_test_predictions.csv"
    rows = read_csv(pred_path)
    y = [int(r["true_label"]) for r in rows]
    p = [float(r["predicted_probability"]) for r in rows]
    pred = [int(float(v) >= 0.5) for v in p]
    out = RESULTS_DIR / "plots"
    out.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y, p)
    plt.tight_layout()
    plt.savefig(out / f"{args.experiment_name}_roc.png", dpi=160)
    plt.close()
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=["healthy", "anaemic"]).plot()
    plt.tight_layout()
    plt.savefig(out / f"{args.experiment_name}_confusion_matrix.png", dpi=160)
    plt.close()

    hist = RESULTS_DIR / "logs" / f"{args.experiment_name}_history.csv"
    if Path(hist).exists():
        rows = read_csv(hist)
        epochs = [int(r["epoch"]) for r in rows]
        for key in ("train_loss", "total_loss", "cbc_recon_loss", "img_recon_loss", "kl_loss", "cls_loss", "f1", "recall"):
            if key in rows[0]:
                vals = [float(r[key]) for r in rows]
                plt.plot(epochs, vals)
                plt.xlabel("epoch")
                plt.ylabel(key)
                plt.tight_layout()
                plt.savefig(out / f"{args.experiment_name}_{key}.png", dpi=160)
                plt.close()
    print(f"Wrote plots for {args.experiment_name} to {out}")


if __name__ == "__main__":
    main()
