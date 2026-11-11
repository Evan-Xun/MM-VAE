# Poster production runbook

## Delivered files

- `MMVAE_Anemia_Screening_Poster_A1_Portrait.pptx`: editable A1 portrait PowerPoint poster.
- `poster.yaml`: style and content manifest for future Paper2Poster-oriented work.
- This folder's three Markdown files: brief, copy, and layout specification.

## Before submission

1. Replace the QR placeholder with the project, paper, or contact URL.
2. Add venue and institutional logos only when permissions and brand guidance are available.
3. Confirm the conference's exact dimensions, bleed, font embedding, and PDF export requirements.
4. Have a clinical co-author check the auxiliary-screening wording and any deployment claims.
5. Export the approved PPTX to PDF and inspect it at final print scale.

## Optional Paper2Poster setup

If using the upstream Paper2Poster repository, place the source PDF and `poster.yaml` together:

```text
{dataset_dir}/mmvae_anemia/paper.pdf
{dataset_dir}/mmvae_anemia/poster.yaml
```

Example command (not executed here; choose installed model names and paths for your environment):

```bash
python -m PosterAgent.new_pipeline \
  --poster_path="{dataset_dir}/mmvae_anemia/paper.pdf" \
  --poster_width_inches=23.39 \
  --poster_height_inches=33.11 \
  --max_workers=3
```

The editable PPTX supplied with this package is the preferred starting point for manual conference customization.
