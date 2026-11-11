import argparse

import torch

from models.mmvvae import MMVAEFromEmbedding
from train_utils import collect_predictions, device, make_loader, prepare_cbc, save_outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--embeddings", default="data/processed/image_embeddings/resnet18_mean.csv")
    p.add_argument("--mode", default="complete", choices=["complete", "missing_image", "missing_cbc"])
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    feature_set = ckpt.get("feature_set", "core_without_hgb")
    pooling = ckpt.get("pooling", "mean")
    latent_dim = int(ckpt.get("latent_dim", 16))
    use_missing_mask = bool(ckpt.get("use_missing_mask", not ckpt_args.get("disable_missing_mask", True)))
    scaler, impute_values, cbc_dim, _ = prepare_cbc(feature_set)
    loader = make_loader("test", feature_set, args.batch_size, False, scaler, impute_values, cbc=True, load_images=False, image_embeddings_csv=args.embeddings)
    dim = int(ckpt.get("image_embedding_dim", loader.dataset.image_embedding_dim))
    model = MMVAEFromEmbedding(cbc_dim, dim, latent_dim, use_missing_mask=use_missing_mask)
    model.load_state_dict(ckpt["model_state"])
    dev = device()
    model.to(dev)
    case_ids, y, prob = collect_predictions(model, loader, dev, "mmvvae", mode=args.mode)
    image_input = {"complete": "12 patches", "missing_image": "Missing image", "missing_cbc": "12 patches"}[args.mode]
    out_feature = "Missing CBC" if args.mode == "missing_cbc" else feature_set
    print(save_outputs(args.experiment_name, "MM-VAE", out_feature, image_input, pooling, latent_dim, case_ids, y, prob))


if __name__ == "__main__":
    main()
