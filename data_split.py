import argparse
import os
from glob import glob

import numpy as np
from sklearn.model_selection import KFold


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create leakage-safe cross-validation splits by grouping files from the same sample/patient."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing feature files.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save split npy files.")
    parser.add_argument("--pattern", type=str, default="*", help="Glob pattern for files under data_dir, e.g. '*.h5'.")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group_level",
        type=str,
        choices=["patient", "sample", "file"],
        default="patient",
        help="Grouping level. For TCGA-style names, patient uses the first 12 chars and sample uses the first 16 chars.",
    )
    parser.add_argument(
        "--prefix_format",
        type=str,
        choices=["stem", "name"],
        default="stem",
        help="Save prefixes as filename stems or full filenames.",
    )
    return parser.parse_args()


def get_prefix(path, prefix_format):
    name = os.path.basename(path)
    if prefix_format == "stem":
        return os.path.splitext(name)[0]
    return name


def get_group_id(prefix, group_level):
    if group_level == "patient":
        return prefix[:12] if prefix.startswith("TCGA") else prefix
    if group_level == "sample":
        return prefix[:16] if prefix.startswith("TCGA") else prefix
    return prefix


def build_grouped_splits(prefixes, group_ids, n_splits, seed):
    unique_groups = np.array(sorted(set(group_ids)))
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"n_splits={n_splits} is larger than the number of unique groups ({len(unique_groups)})."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    kf = KFold(n_splits=n_splits, shuffle=False)

    splits = []
    group_ids = np.array(group_ids)
    prefixes = np.array(prefixes)
    for train_group_idx, val_group_idx in kf.split(unique_groups):
        train_groups = set(unique_groups[train_group_idx])
        val_groups = set(unique_groups[val_group_idx])
        assert train_groups.isdisjoint(val_groups)

        train_mask = np.array([group_id in train_groups for group_id in group_ids])
        val_mask = np.array([group_id in val_groups for group_id in group_ids])
        train_prefix = sorted(prefixes[train_mask].tolist())
        val_prefix = sorted(prefixes[val_mask].tolist())
        splits.append((train_prefix, val_prefix))
    return splits


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    paths = sorted(glob(os.path.join(args.data_dir, args.pattern)))
    paths = [path for path in paths if os.path.isfile(path)]
    if not paths:
        raise FileNotFoundError(f"No files matched {os.path.join(args.data_dir, args.pattern)}")

    prefixes = [get_prefix(path, args.prefix_format) for path in paths]
    group_ids = [get_group_id(prefix, args.group_level) for prefix in prefixes]
    splits = build_grouped_splits(prefixes, group_ids, args.n_splits, args.seed)

    for i, (train_prefix, val_prefix) in enumerate(splits, start=1):
        train_groups = {get_group_id(prefix, args.group_level) for prefix in train_prefix}
        val_groups = {get_group_id(prefix, args.group_level) for prefix in val_prefix}
        assert train_groups.isdisjoint(val_groups), "Train/validation groups overlap."

        save_path = os.path.join(args.save_dir, f"folder_{i}.npy")
        np.save(save_path, np.array([train_prefix, val_prefix], dtype=object))
        print(
            f"Saved {save_path}: "
            f"{len(train_prefix)} train files, {len(val_prefix)} validation files, "
        )


if __name__ == "__main__":
    main()