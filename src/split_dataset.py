import argparse
import random
from collections import defaultdict
from pathlib import Path

from config import MANIFEST_COLUMNS
from utils import read_csv, write_csv


def stratified_split(rows, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label_id"]].append(row)
    split_rows = []
    for _, group in sorted(by_label.items()):
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        for idx, row in enumerate(group):
            row = dict(row)
            row["split"] = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
            split_rows.append(row)
    return sorted(split_rows, key=lambda r: (r["case_id"], r["label_id"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--cbc", default="data/processed/cbc_features.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    rows = stratified_split(read_csv(args.manifest), args.train_ratio, args.val_ratio, args.seed)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "manifest.csv", rows, MANIFEST_COLUMNS)
    for split in ("train", "val", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        write_csv(output_dir / f"{split}.csv", split_rows, MANIFEST_COLUMNS)
        counts = {label: sum(1 for r in split_rows if r["label"] == label) for label in ("healthy", "anaemic")}
        print(f"{split}: {len(split_rows)} rows {counts}")


if __name__ == "__main__":
    main()
