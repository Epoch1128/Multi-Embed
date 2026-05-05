import argparse
import os
import pickle as pkl
import random

import numpy as np
import pandas as pd
import torch

from src import Multi_Embed


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_pickle(path):
    with open(path, "rb") as file:
        return pkl.load(file)


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pkl.dump(obj, file)


def build_st_training_features(image_feature_path, rna_path):
    """Align spot-level image features and RNA features by spot ID."""
    image_features = load_pickle(image_feature_path)
    rna_df = pd.read_csv(rna_path, sep="\t", index_col=0)

    img_feats, rna_feats, spot_ids = [], [], []
    for key, value in image_features.items():
        spot_id = key.split("_")[3]
        if spot_id not in rna_df.index:
            continue
        img_feats.append(value)
        rna_feats.append(np.array(rna_df.loc[spot_id]))
        spot_ids.append(spot_id)

    if not spot_ids:
        raise ValueError("No matched spots were found between image features and RNA table.")

    feature_dict = {
        "image": np.stack(img_feats),
        "rna": np.stack(rna_feats),
    }
    return feature_dict, spot_ids


def train_model(feature_dict, sample_names, args):
    model_configs = {
        "hidden_size": args.hidden_size,
        "embed_size": args.embed_size,
        "checkpoints": args.checkpoint,
    }
    exp = Multi_Embed(
        model_configs,
        features=feature_dict,
        anchor_keys="image",
        sample_names=sample_names,
        device=args.device,
    )
    exp.train(epochs=args.epochs, learning_rate=args.learning_rate, beta=args.beta)
    return exp


def get_image_embeddings(exp, image_feature_path):
    image_features = load_pickle(image_feature_path)
    image_names = list(image_features.keys())
    test_images = np.stack(list(image_features.values()))
    embed_dict = exp.get_embeddings({"image": test_images})
    return embed_dict, image_names


def parse_spatial_coords(sample_names):
    coords = []
    for sample_name in sample_names:
        x, y = sample_name.split("_")[-1].split("x")
        coords.append([int(x), int(y)])
    return np.array(coords)


def visualize_embeddings(embeddings, sample_names, mask_path, save_dir, cluster_num):
    import cv2
    from tnbc_plot import visual_tnbc

    coords = parse_spatial_coords(sample_names)
    mask = cv2.imread(mask_path, flags=0)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found or cannot be read: {mask_path}")
    visual_tnbc(embeddings["image"], coords, mask, save_dir=save_dir, cluster_num=cluster_num)


def run_pipeline(args):
    setup_seed(args.seed)
    print("Loading training data...")
    feature_dict, spot_ids = build_st_training_features(args.train_image_features, args.rna_path)

    print("Training Multi-Embed...")
    exp = train_model(feature_dict, spot_ids, args)

    print("Generating image embeddings...")
    embed_dict, image_names = get_image_embeddings(exp, args.test_image_features)

    output_path = os.path.join(args.save_dir, args.embedding_save_name)
    save_pickle(
        {
            "embeddings": embed_dict,
            "sample_names": image_names,
            "train_spot_ids": spot_ids,
        },
        output_path,
    )
    print(f"Embeddings saved to {output_path}")

    if args.visualize:
        if args.mask_path is None:
            raise ValueError("--mask-path is required when --visualize is enabled.")
        visualize_embeddings(
            embed_dict,
            image_names,
            args.mask_path,
            args.save_dir,
            args.cluster_num,
        )

    return embed_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Multi-Embed on paired ST image/RNA features and export image embeddings."
    )
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # Take TNBC data as example, you can change the data path to your own data
    # Also, by changing the train-image-features and rna-path or add another modality, you can train the model in multi-omics pattern.
    parser.add_argument("--train-image-features", type=str, default="TNBC/CN15_D2.pkl")
    parser.add_argument("--rna-path", type=str, default="TNBC/CN15_D2.tsv")
    
    # This script is a demo for super-resolution clustering, you can change the input test data to get other embeddings.
    parser.add_argument("--test-image-features", type=str, default="TNBC/CN15_D2_sr_112.pkl")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--embedding-save-name", type=str, default="st_embeddings.pkl")

    # Model hyperparameters
    parser.add_argument("--hidden-size", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--embed-size", type=int, default=512)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0)

    # Visualization hyperparameters
    parser.add_argument("--visualize", action="store_true", help="Run cluster visualization after embedding.")
    parser.add_argument("--cluster-num", type=int, default=6)
    parser.add_argument("--mask-path", type=str, default=None)  # The tissue mask
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()