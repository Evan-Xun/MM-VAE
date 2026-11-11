import argparse
import re
from pathlib import Path

from config import DEFAULT_DATA_ROOT, LABEL_ALIASES, LABELS, MANIFEST_COLUMNS
from utils import write_csv


def find_file(folder, pattern):
    matches = sorted(Path(folder).glob(pattern))
    return str(matches[0]) if matches else ""


def collect_case(data_root, class_dir):
    label, suffix = LABEL_ALIASES[class_dir.name]
    label_id = LABELS[label]
    original_dir = class_dir / "Original_images"
    cbc_dir = class_dir / "CBC_reports"
    morph_dir = class_dir / "Morphology_reports"
    patch_dir = data_root / "AneRBC-II" / class_dir.name / "Original_images"

    rows = []
    for original in sorted(original_dir.glob(f"*_{suffix}.png")):
        match = re.match(r"(\d{3})_([ah])\.png$", original.name)
        if not match:
            continue
        case_num = match.group(1)
        case_id = f"{case_num}_{suffix}"
        patch_prefix = f"{int(case_num):04d}"
        patches = [find_file(patch_dir, f"{patch_prefix}_{i:02d}_{suffix}.png") for i in range(1, 13)]
        row = {
            "case_id": case_id,
            "label": label,
            "label_id": str(label_id),
            "original_image_path": str(original),
            "cbc_report_path": str(cbc_dir / f"{case_id}.txt"),
            "morphology_report_path": str(morph_dir / f"{case_id}.txt"),
            "split": "",
        }
        row.update({f"patch_{i}_path": patches[i - 1] for i in range(1, 13)})
        rows.append(row)
    return rows


def build_manifest(data_root):
    data_root = Path(data_root)
    rows = []
    for class_name in ("Healthy_individuals", "Anemic_individuals"):
        rows.extend(collect_case(data_root, data_root / "AneRBC-I" / class_name))
    missing = [r["case_id"] for r in rows if any(not r[f"patch_{i}_path"] for i in range(1, 13))]
    if missing:
        raise RuntimeError(f"{len(missing)} cases do not have 12 patches, first missing: {missing[:5]}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output", default="data/processed/manifest.csv")
    args = parser.parse_args()
    rows = build_manifest(args.data_root)
    write_csv(args.output, rows, MANIFEST_COLUMNS)
    counts = {label: sum(1 for r in rows if r["label"] == label) for label in LABELS}
    print(f"Wrote {len(rows)} patient-level rows to {args.output}: {counts}")


if __name__ == "__main__":
    main()
