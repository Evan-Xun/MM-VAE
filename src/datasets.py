from pathlib import Path

from config import FEATURE_SETS
from utils import read_csv


class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, values):
        import numpy as np
        arr = np.asarray(values, dtype="float32")
        self.mean = arr.mean(axis=0)
        self.std = arr.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, values):
        import numpy as np
        return (np.asarray(values, dtype="float32") - self.mean) / self.std


def load_cbc_table(path, feature_set):
    features = FEATURE_SETS[feature_set]
    table = {}
    masks = {}
    for row in read_csv(path):
        vals = []
        mask = []
        for name in features:
            value = row.get(name, "")
            observed = value not in ("", "nan", "None")
            vals.append(float(value) if observed else float("nan"))
            mask.append(1.0 if observed else 0.0)
        table[row["case_id"]] = vals
        masks[row["case_id"]] = mask
    return table, masks, features


def impute_by_train(cbc_by_case, train_case_ids):
    import numpy as np
    train = np.asarray([cbc_by_case[cid] for cid in train_case_ids], dtype="float32")
    med = np.nanmedian(train, axis=0)
    filled = {}
    for cid, vals in cbc_by_case.items():
        arr = np.asarray(vals, dtype="float32")
        arr = np.where(np.isnan(arr), med, arr)
        filled[cid] = arr.tolist()
    return filled, med


def build_transform(train, image_size=224):
    from torchvision import transforms
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(ops)


class PatientLevelDataset:
    def __init__(self, manifest_csv, cbc_features_csv=None, feature_set="core_without_hgb",
                 train=False, scaler=None, impute_values=None, image_size=224, load_images=True,
                 image_embeddings_csv=None):
        import torch
        self.torch = torch
        self.rows = read_csv(manifest_csv)
        self.load_images = load_images
        self.transform = build_transform(train=train, image_size=image_size) if load_images else None
        self.image_embeddings = None
        self.image_embedding_dim = 0
        if image_embeddings_csv:
            self.image_embeddings = {}
            for row in read_csv(image_embeddings_csv):
                values = [float(row[f"emb_{i}"]) for i in range(int(row["embedding_dim"]))]
                self.image_embeddings[row["case_id"]] = values
            self.image_embedding_dim = len(next(iter(self.image_embeddings.values()))) if self.image_embeddings else 0
        self.cbc_by_case = None
        self.cbc_masks_by_case = None
        self.feature_names = []
        if cbc_features_csv:
            self.cbc_by_case, self.cbc_masks_by_case, self.feature_names = load_cbc_table(cbc_features_csv, feature_set)
            if impute_values is not None:
                import numpy as np
                for cid, vals in self.cbc_by_case.items():
                    arr = np.asarray(vals, dtype="float32")
                    self.cbc_by_case[cid] = np.where(np.isnan(arr), impute_values, arr).tolist()
            if scaler is not None:
                for cid, vals in self.cbc_by_case.items():
                    self.cbc_by_case[cid] = scaler.transform([vals])[0].tolist()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        from PIL import Image
        row = self.rows[idx]
        item = {
            "case_id": row["case_id"],
            "label": self.torch.tensor(float(row["label_id"]), dtype=self.torch.float32),
        }
        if self.load_images:
            patches = []
            for i in range(1, 13):
                with Image.open(Path(row[f"patch_{i}_path"])) as img:
                    patches.append(self.transform(img.convert("RGB")))
            item["patches"] = self.torch.stack(patches, dim=0)
        if self.cbc_by_case is not None:
            item["cbc"] = self.torch.tensor(self.cbc_by_case[row["case_id"]], dtype=self.torch.float32)
            item["cbc_mask"] = self.torch.tensor(self.cbc_masks_by_case[row["case_id"]], dtype=self.torch.float32)
        if self.image_embeddings is not None:
            item["image_embedding"] = self.torch.tensor(self.image_embeddings[row["case_id"]], dtype=self.torch.float32)
        return item
