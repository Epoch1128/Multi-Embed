import numpy as np
import torch
import random
import os
from typing import Optional

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(model, optimizer, save_dir):
    print(f"Saving checkpoint to {save_dir}")
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_dir)

def collate_func(batch):
    image_feat = torch.cat([item['image_feat'] for item in batch], dim = 0)
    neg_image_feat = torch.cat([item['neg_feat'] for item in batch], dim = 0)
    omic_feat = torch.cat([item['omic_feat'] for item in batch], dim = 0)
    names = [item['name'] for item in batch]
    return {
        'name': names,
        'image_feat': image_feat,
        'omic_feat': omic_feat,
        'neg_feat': neg_image_feat
    }

def get_prefix(prefix):
    prefix_pth = prefix[0]
    if os.path.exists(prefix_pth):
        return list(np.load(prefix_pth, allow_pickle=True)[0])
    if isinstance(prefix, list):
        return prefix  
    else:
        return [prefix]
    

def top_k_recall(predictions, labels, k):
    # assert predictions.shape[0] == labels.shape[0], "length of predictions and labels should be the same"

    correct_predictions = 0
    total_labels = np.unique(labels).shape[0]

    for i in range(total_labels):
        if labels[i] in predictions[i, :k]:
            correct_predictions += 1

    recall = correct_predictions / total_labels
    return recall


def top_k_recall_tiles(predictions, labels, label_index, k):
    correct_predictions = 0
    total_labels = np.unique(labels).shape[0]

    for i in range(total_labels):
        if labels[i] in label_index[list(predictions[i, :k])]:
            correct_predictions += 1

    recall = correct_predictions / total_labels
    return recall


def calc_recall(norm_omics_mat, norm_images, topk_list=[1, 5, 10]):
    if isinstance(norm_images, list):
        flatten_idx = np.concatenate([np.full(n.shape[0], i) for i, n in enumerate(norm_images)])    # all images
        all_norm_images = np.concatenate(norm_images)
        cosine_similarity_matrix = np.matmul(norm_omics_mat, all_norm_images.T)
        retrival_idx = np.argsort(-cosine_similarity_matrix, axis=-1)   # gene x all images
        labels = np.arange(norm_omics_mat.shape[0])   # gene
        recall_dict = {}
        for k in topk_list:
            recall = top_k_recall_tiles(retrival_idx, labels, flatten_idx, k=k)
            recall_dict.update(
                {
                    str(k): recall
                }
            )

    elif isinstance(norm_images, np.ndarray):
        cosine_similarity_matrix = np.matmul(norm_omics_mat, norm_images.T) # gene x gene
        retrival_idx = np.argsort(-cosine_similarity_matrix, axis=-1)
        labels = np.arange(norm_omics_mat.shape[0])   # gene
        recall_dict = {}
        for k in topk_list:
            recall = top_k_recall(retrival_idx, labels, k=k)
            recall_dict.update(
                {
                    str(k): recall
                }
            )

    else:
        raise NotImplementedError
    return recall_dict

def normalize(input: np.ndarray, p: float = 2.0, dim: int = 1, eps: float = 1e-12, out: Optional[np.ndarray] = None) -> np.ndarray:
    norm = np.linalg.norm(input, ord=p, axis=dim, keepdims=True)
    denom = np.maximum(norm, eps)
    
    if out is None:
        return input / denom
    else:
        np.divide(input, denom, out=out)
        return out