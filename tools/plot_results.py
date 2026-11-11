"""
Generate result figures from paper tables (Table 4-1, 4-2, 4-3).
Outputs are saved to artifacts/.
"""

import matplotlib
matplotlib.use("Agg")
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "../artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "NPV", "F1", "ROC-AUC"]

# ── Colour palette ──────────────────────────────────────────────────────────
COLORS = {
    "Rule-based":     "#9E9E9E",
    "CBC-only":       "#5C85D6",
    "Image-only":     "#F4A460",
    "Concat Fusion":  "#66BB6A",
    "MM-VAE":         "#EF5350",
    "with HGB":       "#5C85D6",
    "without HGB":    "#EF5350",
    "complete":       "#5C85D6",
    "missing_image":  "#66BB6A",
    "missing_cbc":    "#EF5350",
    "NPV":            "#2E7D32",
    "F1":             "#1565C0",
    "ROC-AUC":        "#C62828",
}

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Figure 1: Table 4-1 – Main Comparison (core_with_hgb) ──────────────────
def plot_main_comparison():
    data = {
        "Rule-based":    [0.5933, 0.5641, 0.8800, 0.3200, 0.6400, 0.6875, None],
        "CBC-only":      [0.7000, 0.6829, 0.7467, 0.6533, 0.7206, 0.7134, 0.7378],
        "Image-only":    [0.5000, 0.5000, 1.0000, 0.0000, 0.0000, 0.6667, 0.6204],
        "Concat Fusion": [0.7520, 0.7342, 0.7973, 0.7067, 0.7805, 0.7625, 0.8361],
        "MM-VAE":        [0.7773, 0.7578, 0.8160, 0.7387, 0.8011, 0.7856, 0.8561],
    }
    errors = {
        "Concat Fusion": [0.0213, 0.0395, 0.0561, 0.0742, 0.0372, 0.0190, 0.0106],
        "MM-VAE":        [0.0060, 0.0131, 0.0219, 0.0242, 0.0155, 0.0065, 0.0210],
    }

    n_models = len(data)
    n_metrics = len(METRICS)
    x = np.arange(n_metrics)
    width = 0.15
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(13, 5.8))

    for i, (model, vals) in enumerate(data.items()):
        yerr = errors.get(model, [None] * n_metrics)
        plot_vals = [v if v is not None else 0 for v in vals]
        err_arr = [e if e is not None else 0 for e in yerr]
        bars = ax.bar(
            x + offsets[i], plot_vals, width,
            label=model,
            color=COLORS[model],
            yerr=err_arr if any(e > 0 for e in err_arr) else None,
            capsize=3, error_kw={"linewidth": 1},
            zorder=3,
        )
        # Mark N/A bar (Rule-based ROC-AUC)
        if model == "Rule-based":
            ax.text(x[-1] + offsets[i], 0.02, "N/A",
                    ha="center", va="bottom", fontsize=8, color="white",
                    rotation=90, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=5,
        fontsize=13,
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig_table4_1_main_comparison.png")


# ── Figure 2: Table 4-2 – HGB Ablation ────────────────────────────────────
def plot_hgb_ablation():
    models = ["CBC-only", "Concat Fusion", "MM-VAE"]
    with_hgb = {
        "CBC-only":      [0.7000, 0.6829, 0.7467, 0.6533, 0.7206, 0.7134, 0.7378],
        "Concat Fusion": [0.7467, 0.7229, 0.8000, 0.6933, 0.7761, 0.7595, 0.8339],
        "MM-VAE":        [0.7800, 0.7442, 0.8533, 0.7067, 0.8281, 0.7950, 0.8390],
    }
    without_hgb = {
        "CBC-only":      [0.5800, 0.7143, 0.2667, 0.8933, 0.5492, 0.3883, 0.7230],
        "Concat Fusion": [0.7733, 0.7662, 0.7867, 0.7600, 0.7808, 0.7763, 0.8063],
        "MM-VAE":        [0.7733, 0.7253, 0.8800, 0.6667, 0.8475, 0.7952, 0.8234],
    }

    n_metrics = len(METRICS)
    x = np.arange(n_metrics)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, model in zip(axes, models):
        w = with_hgb[model]
        wo = without_hgb[model]
        width = 0.35
        ax.bar(x - width / 2, w,  width, label="with HGB",    color=COLORS["with HGB"],    zorder=3)
        ax.bar(x + width / 2, wo, width, label="without HGB", color=COLORS["without HGB"], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(METRICS, rotation=30, ha="right", fontsize=13)
        ax.set_ylim(0, 1.12)
        ax.set_title(model, fontweight="bold", fontsize=14)
        ax.tick_params(axis="y", labelsize=13)
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", zorder=1)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(fontsize=13)

    axes[0].set_ylabel("Score", fontsize=15)
    fig.tight_layout()
    save(fig, "fig_table4_2_hgb_ablation.png")


# ── Figure 3: Table 4-3 – Missing-Modality Robustness ─────────────────────
def plot_missing_modality():
    data = {
        "complete":      [0.7800, 0.7442, 0.8533, 0.7067, 0.8281, 0.7950, 0.8390],
        "missing_image": [0.7000, 0.6705, 0.7867, 0.6133, 0.7419, 0.7239, 0.7262],
        "missing_cbc":   [0.5333, 0.5472, 0.3867, 0.6800, 0.5258, 0.4531, 0.5892],
    }

    n_modes = len(data)
    x = np.arange(len(METRICS))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(11, 4.5))

    for offset, (mode, vals) in zip(offsets, data.items()):
        ax.bar(x + offset, vals, width, label=mode, color=COLORS[mode], zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    save(fig, "fig_table4_3_missing_modality.png")


def plot_missing_rate_simulation():
    path = os.path.join(
        os.path.dirname(__file__),
        "../results/metrics/missing_rate_simulation_with_hgb_summary.csv",
    )
    if not os.path.exists(path):
        print(f"Skipped missing-rate simulation figure; not found: {path}")
        return

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    metric_map = {
        "NPV": "npv_mean",
        "F1": "f1_mean",
        "ROC-AUC": "roc_auc_mean",
    }
    scenario_titles = {
        "image_missing": "Increasing image missingness",
        "cbc_missing": "Increasing CBC missingness",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, scenario in zip(axes, ["image_missing", "cbc_missing"]):
        subset = [r for r in rows if r["scenario"] == scenario]
        subset.sort(key=lambda r: float(r["missing_rate"]))
        x = [float(r["missing_rate"]) * 100 for r in subset]
        for label, key in metric_map.items():
            y = [float(r[key]) for r in subset]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=5,
                label=label,
                color=COLORS[label],
            )
        ax.set_title(scenario_titles[scenario], fontweight="bold")
        ax.set_xlabel("Missing samples (%)")
        ax.set_xticks(x)
        ax.set_ylim(0.4, 0.9)
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", zorder=1)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel("Score")
    fig.tight_layout()
    save(fig, "fig_missing_rate_simulation.png")


if __name__ == "__main__":
    plot_main_comparison()
    plot_hgb_ablation()
    plot_missing_modality()
    plot_missing_rate_simulation()
    print("All figures saved to artifacts/")
