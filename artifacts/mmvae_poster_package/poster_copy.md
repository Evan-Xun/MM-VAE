# Poster copy

## Title block

**A Multimodal Variational Autoencoder for Anemia Screening via CBC Indicators and Blood Smear Imagery**

Fengxun Liu, Yuxuan Huang, Zeyuan Li, Xiaolin Li*

School of Computing and Data Science, Xiamen University Malaysia, Sepang, Malaysia; Department of Clinical Laboratory, The Second Affiliated Hospital of Nanchang University, Nanchang, China.  
*Corresponding author

## Why this matters

**CBC**: strong quantitative signal -> may over-rely on HGB.  
**Blood smear**: morphological evidence -> weak alone in the current representation.  
**Existing fusion**: simple concatenation -> no explicit uncertainty modelling.

**Can probabilistic fusion combine both sources more robustly?**

## Our approach

**Patient-level MM-VAE with Product-of-Experts (PoE) fusion**

- One case: one CBC report + 12 RBC patches + one binary anemia label.
- Frozen ResNet-18 creates 512-D patch embeddings; mean pooling forms a patient-level image embedding.
- CBC features plus missingness mask and image embeddings are encoded as modality-specific Gaussian posteriors.
- PoE fuses posteriors into a 16-D shared latent variable for reconstruction and anemia classification.

**Why PoE instead of concatenation?** Precision-weighted posterior fusion lets each modality contribute according to its estimated uncertainty and naturally supports missing-modality inference.

## Experimental setup

- AneRBC dataset; patient-level stratified split: 700 / 150 / 150 train / validation / test cases.
- CBC: RBC, HGB, HCT, MCV, MCH, MCHC, RDW-CV; training-set median imputation, z-score normalization, and missingness mask.
- Training: Adam, learning rate 0.001, batch size 32, <=100 epochs, early stopping (15), KL warm-up (20 epochs).
- Compared rule-based HGB, CBC-only, image-only, Concat Fusion, and MM-VAE; five seeds for main comparison.

## Key result: five-seed means

**MM-VAE achieved the best overall performance across five random seeds:** F1 **0.786 +/- 0.007**, ROC-AUC **0.856 +/- 0.021**, and NPV **0.801 +/- 0.016**.

Concat Fusion achieved F1 **0.763 +/- 0.020**, ROC-AUC **0.836 +/- 0.009**, and NPV **0.781 +/- 0.037**. The poster does not claim formal statistical significance.

## HGB ablation

Removing HGB tests whether the model relies on a single dominant indicator. Values below are from the representative HGB-ablation run.

- CBC-only F1: **0.7134 -> 0.3883**; recall: **0.7467 -> 0.2667**.
- MM-VAE F1: **0.7950 -> 0.7952**; ROC-AUC: **0.8390 -> 0.8234**.
- **Key finding:** MM-VAE remains stable without HGB, supporting complementary use of remaining CBC indicators and image morphology.

## Missing-modality robustness

The same trained MM-VAE supports inference without retraining.

- With the image modality missing: accuracy 0.7000, F1 0.7239, ROC-AUC 0.7262, NPV 0.7419.
- With the CBC modality missing: accuracy 0.5333, F1 0.4531, ROC-AUC 0.5892, NPV 0.5258.
- CBC is the stronger expert; imagery adds complementary evidence and stability when available.

## Potential clinical role

Use MM-VAE as an **auxiliary screening tool**: reuse routine CBC from the laboratory information system, add smear embeddings when microscopy is available, and route probability outputs to clinician-led triage.

**Takeaway:** PoE-based latent fusion outperforms direct concatenation and reduces dependence on HGB, while making the system’s modality asymmetry explicit.

## Limitations / next steps

- Single public dataset.
- Frozen image encoder.
- No prospective clinical validation.
- Next: multi-center validation, stronger image encoder, and threshold calibration.
