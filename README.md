# MM-VAE Multimodal Anemia Detection

This project implements a patient-level experimental pipeline for binary anemia screening with the AneRBC dataset. One sample is one original case: one CBC report, twelve AneRBC-II image patches, and one label. The split is always performed before patch loading, so patches from the same original image never cross train, validation, and test boundaries.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The checked-in data path is auto-detected by default. To use another copy, pass `--data-root /path/to/AneRBC_dataset`.

## Preprocess

```bash
python src/build_manifest.py \
  --data-root "AneRBC dataset a benchmark dataset for computer-aided anemia diagnosis using RBC images. httpsdoi.org10.1093databasebaae120/AneRBC_dataset" \
  --output data/processed/manifest.csv

python src/parse_cbc.py \
  --manifest data/processed/manifest.csv \
  --output data/processed/cbc_features.csv

python src/split_dataset.py \
  --manifest data/processed/manifest.csv \
  --cbc data/processed/cbc_features.csv \
  --output-dir data/processed \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

Expected split: 700 train, 150 validation, 150 test, balanced by class.

## Experiments

On CPU-only machines, first cache frozen ResNet18 patch embeddings. This keeps the patient-level logic identical while avoiding repeated CNN forward passes during every training epoch.

```bash
python src/extract_image_embeddings.py --backbone resnet18 --pooling mean --pretrained --batch-size 16 --output data/processed/image_embeddings/resnet18_mean.csv
```

Rule-based clinical baseline:

```bash
python src/rule_based_baseline.py --threshold 13.0 --experiment-name G0_rule_hgb_lt_13
```

CBC-only baselines:

```bash
python src/train_cbc_baseline.py --feature-set core_with_hgb --experiment-name G1_cbc_core_with_hgb
python src/train_cbc_baseline.py --feature-set core_without_hgb --experiment-name G2_cbc_core_without_hgb
```

Image-only baseline:

```bash
python src/train_image_baseline.py --backbone resnet18 --pooling mean --experiment-name G3_image_only_resnet18_mean
python src/train_image_embedding_baseline.py --embeddings data/processed/image_embeddings/resnet18_mean.csv --pooling mean --experiment-name G3_image_only_resnet18_mean
```

Late fusion:

```bash
python src/train_late_fusion.py --feature-set core_without_hgb --backbone resnet18 --pooling mean --experiment-name G4_late_fusion_without_hgb
python src/train_late_fusion.py --feature-set core_with_hgb --backbone resnet18 --pooling mean --experiment-name G6_late_fusion_with_hgb
python src/train_late_fusion_embedding.py --feature-set core_without_hgb --embeddings data/processed/image_embeddings/resnet18_mean.csv --pooling mean --experiment-name G4_late_fusion_without_hgb
python src/train_late_fusion_embedding.py --feature-set core_with_hgb --embeddings data/processed/image_embeddings/resnet18_mean.csv --pooling mean --experiment-name G6_late_fusion_with_hgb
```

MM-VAE:

```bash
python src/train_mmvvae.py --feature-set core_without_hgb --backbone resnet18 --pooling mean --latent-dim 16 --kl-warmup-epochs 20 --experiment-name G5_mmvvae_without_hgb
python src/train_mmvvae.py --feature-set core_with_hgb --backbone resnet18 --pooling mean --latent-dim 16 --kl-warmup-epochs 20 --experiment-name G7_mmvvae_with_hgb
python src/train_mmvvae_embedding.py --feature-set core_without_hgb --embeddings data/processed/image_embeddings/resnet18_mean.csv --pooling mean --latent-dim 16 --kl-warmup-epochs 20 --alpha 0.1 --beta 0.1 --lambda-cls 10 --experiment-name G5_mmvvae_without_hgb
python src/train_mmvvae_embedding.py --feature-set core_with_hgb --embeddings data/processed/image_embeddings/resnet18_mean.csv --pooling mean --latent-dim 16 --kl-warmup-epochs 20 --alpha 0.1 --beta 0.1 --lambda-cls 10 --experiment-name G7_mmvvae_with_hgb
```

Missing-modality evaluation:

```bash
python src/evaluate.py --checkpoint results/checkpoints/G5_mmvvae_without_hgb_best.pt --model mmvvae --mode complete --experiment-name M1_mmvvae_complete
python src/evaluate.py --checkpoint results/checkpoints/G5_mmvvae_without_hgb_best.pt --model mmvvae --mode missing_image --experiment-name M2_mmvvae_missing_image
python src/evaluate.py --checkpoint results/checkpoints/G5_mmvvae_without_hgb_best.pt --model mmvvae --mode missing_cbc --experiment-name M3_mmvvae_missing_cbc
python src/evaluate_mmvvae_embedding.py --checkpoint results/checkpoints/G5_mmvvae_without_hgb_best.pt --embeddings data/processed/image_embeddings/resnet18_mean.csv --mode complete --experiment-name M1_mmvvae_complete
python src/evaluate_mmvvae_embedding.py --checkpoint results/checkpoints/G5_mmvvae_without_hgb_best.pt --embeddings data/processed/image_embeddings/resnet18_mean.csv --mode missing_image --experiment-name M2_mmvvae_missing_image
python src/evaluate_mmvvae_embedding.py --checkpoint results/checkpoints/G5_mmvvae_without_hgb_best.pt --embeddings data/processed/image_embeddings/resnet18_mean.csv --mode missing_cbc --experiment-name M3_mmvvae_missing_cbc
```

Plots:

```bash
python src/plot_results.py --experiment-name G5_mmvvae_without_hgb
```

## Outputs

Metrics are saved to `results/metrics/{experiment_name}_metrics.json`, predictions to `results/predictions/{experiment_name}_test_predictions.csv`, checkpoints to `results/checkpoints/`, and the combined table to `results/metrics/all_results_summary.csv`.

Primary interpretation should compare `G2`, `G3`, `G4`, and `G5` to test whether multimodal fusion improves anemia screening when hemoglobin is excluded. Use `M1`, `M2`, and `M3` to evaluate graceful degradation when one modality is missing.
