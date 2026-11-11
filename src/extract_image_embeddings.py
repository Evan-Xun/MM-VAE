import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import PROCESSED_DIR
from datasets import PatientLevelDataset
from image_encoder import PatchImageEncoder
from train_utils import device
from utils import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--output", default="data/processed/image_embeddings/resnet18_mean.csv")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pooling", default="mean", choices=["mean", "attention"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    dev = device()
    model = PatchImageEncoder(args.backbone, args.pooling, pretrained=args.pretrained, freeze=True).to(dev)
    model.eval()
    output = Path(args.output)
    ensure_dir(output.parent)

    rows = []
    for split in ("train", "val", "test"):
        ds = PatientLevelDataset(Path(args.manifest_dir) / f"{split}.csv", cbc_features_csv=None, train=False, load_images=True)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in loader:
                emb = model(batch["patches"].to(dev)).cpu().numpy()
                for case_id, label, vec in zip(batch["case_id"], batch["label"].numpy().astype(int), emb):
                    rows.append({
                        "case_id": case_id,
                        "label_id": int(label),
                        "split": split,
                        "embedding_dim": emb.shape[1],
                        **{f"emb_{i}": float(v) for i, v in enumerate(vec)},
                    })
    fieldnames = ["case_id", "label_id", "split", "embedding_dim", *[f"emb_{i}" for i in range(model.embedding_dim)]]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} image embeddings to {output}")


if __name__ == "__main__":
    main()
