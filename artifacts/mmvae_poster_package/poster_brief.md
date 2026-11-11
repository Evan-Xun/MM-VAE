# MM-VAE anemia screening poster brief

## Poster goal

Show an academic and clinical-informatics audience that probabilistic fusion of CBC indicators and red blood-cell smear imagery improves patient-level anemia screening, and remains useful when a strong CBC feature or one modality is unavailable.

## Audience and format

- Audience: conference reviewers, clinical informatics researchers, and laboratory-medicine practitioners.
- Format: A1 portrait (594 x 841 mm / 23.39 x 33.11 inches) PowerPoint poster; editable single-slide PPTX.
- Language: English, matching the source manuscript.
- Visual emphasis: PoE architecture as the hero visual; three evidence charts for main comparison, HGB ablation, and gradual modality missingness.

## Main takeaway

**MM-VAE integrates quantitative CBC evidence and morphological image evidence in a shared probabilistic latent space, achieving mean F1 0.786 +/- 0.007 and ROC-AUC 0.856 +/- 0.021 across five seeds on AneRBC.**

## Evidence priorities

1. Across five seeds, MM-VAE has higher mean F1, ROC-AUC, and NPV than Concat Fusion.
2. Removing HGB leaves MM-VAE nearly unchanged (F1 0.7950 to 0.7952), unlike CBC-only.
3. Missing-image performance remains useful, while the sharper missing-CBC drop transparently shows that CBC is the stronger modality.

## Source of truth

- Manuscript: `/Users/fengxun/Downloads/Anemia_Screening_MMVAE1.docx`.
- Experimental metrics and source plots: `results/metrics/` and `results/plots/`.

## Assumptions / open items

- No venue branding, poster template, logo, or QR destination was supplied.
- The footer retains a QR/contact placeholder; replace it before submission.
- The poster presents the system as an auxiliary screening tool, not an independent diagnostic system.
