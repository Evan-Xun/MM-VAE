import argparse

import torch

from models.cbc_baseline import CBCBaseline
from models.image_baseline import ImageBaseline
from models.late_fusion import LateFusion
from models.mmvvae import MMVAE
from train_utils import collect_predictions, device, make_loader, prepare_cbc, save_outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", choices=["cbc", "image", "late_fusion", "mmvvae"], required=True)
    p.add_argument("--mode", default="complete", choices=["complete", "missing_image", "missing_cbc"])
    p.add_argument("--feature-set", default=None)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--pooling", default="mean")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    feature_set = args.feature_set or ckpt.get("feature_set", "core_without_hgb")
    pooling = ckpt.get("pooling", args.pooling)
    latent_dim = int(ckpt.get("latent_dim", args.latent_dim))
    use_missing_mask = bool(ckpt.get("use_missing_mask", not ckpt_args.get("disable_missing_mask", True)))
    scaler, impute_values, cbc_dim, _ = prepare_cbc(feature_set)
    cbc_needed = args.model != "image" and args.mode != "missing_cbc"
    loader = make_loader(
        "test", feature_set, args.batch_size, False, scaler, impute_values,
        cbc=cbc_needed or args.model != "image",
        load_images=args.model in ("image", "late_fusion", "mmvvae") and args.mode != "missing_image",
    )
    if args.model == "cbc":
        model, kind, label = CBCBaseline(cbc_dim, use_missing_mask=use_missing_mask), "cbc", "CBC-only"
    elif args.model == "image":
        model, kind, label = ImageBaseline(args.backbone, pooling), "image", "Image-only"
    elif args.model == "late_fusion":
        model, kind, label = LateFusion(cbc_dim, args.backbone, pooling, use_missing_mask=use_missing_mask), "late", "Late Fusion"
    else:
        model, kind, label = MMVAE(cbc_dim, latent_dim, args.backbone, pooling, use_missing_mask=use_missing_mask), "mmvvae", "MM-VAE"
    dev = device()
    model.load_state_dict(ckpt["model_state"])
    model.to(dev)
    case_ids, y, prob = collect_predictions(model, loader, dev, kind, mode=args.mode)
    image_input = {"complete": "12 patches", "missing_image": "Missing image", "missing_cbc": "12 patches"}[args.mode]
    out_feature = "Missing CBC" if args.mode == "missing_cbc" else feature_set
    print(save_outputs(args.experiment_name, label, out_feature, image_input, pooling, latent_dim, case_ids, y, prob))


if __name__ == "__main__":
    main()
