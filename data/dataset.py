import os
import numpy as np
import pickle as pkl
import pandas as pd
import random
import torch
import h5py
from glob import glob
from torch.utils.data import Dataset

# GENE_LIST = list(np.load('/home/gcf22/workspace/Co-embedding/data/gene_list.npy', allow_pickle=True))

def generate_random_except(except_index, cluster_list, min_val, max_val):
    while True:
        random_num = random.randint(min_val, max_val)
        if random_num != except_index and cluster_list[except_index] != cluster_list[random_num]:
            return random_num
        
def generate_random_except_without_cluster(except_index, min_val, max_val):
    while True:
        random_num = random.randint(min_val, max_val)
        if random_num != except_index:
            return random_num

class STDataset(Dataset):
    def __init__(self, img_dir, omics_dir, prefixs=None):
        if prefixs is None:
            img_pths = glob(os.path.join(img_dir, '*.pkl'))

        else:
            img_pths = [os.path.join(img_dir, f'{prefix}.pkl') for prefix in prefixs]

        name_list, feat_list, omic_list = [], [], []
        for img_pth in img_pths:
            sample_name = img_pth.split('/')[-1].split('.')[0]
            omic_pth = os.path.join(omics_dir, f'{sample_name}.tsv')
            with open(img_pth, 'rb') as file:
                feat_dict = pkl.load(file)
            omic_df = pd.read_csv(omic_pth, sep='\t', index_col=0)
            for key, value in feat_dict.items():
                x, y = key.split('_')[-2].split('x')
                if f'{x}x{y}' not in omic_df.index:
                    print(f'{x}x{y}')
                    continue
                omic_feat = omic_df.loc[f'{x}x{y}']
                omic_list.append(np.array(omic_feat))
                feat_list.append(value.squeeze())
                name_list.append(key)
        self.name_list = name_list
        self.feat_list = feat_list
        self.omic_list = omic_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        neg_idx = generate_random_except_without_cluster(index, 0, len(self.feat_list) - 1)
        return {
            'name': self.name_list[index],
            'image_feat': torch.tensor(self.feat_list[index].squeeze(), dtype=torch.float32).unsqueeze(0),
            'omic_feat': torch.tensor(self.omic_list[index], dtype=torch.float32).unsqueeze(0),
            'neg_feat': torch.tensor(self.feat_list[neg_idx].squeeze(), dtype=torch.float32).unsqueeze(0)
        }
    
class XeniumDataset(Dataset):
    def __init__(self, img_dir, omics_dir, cluster_dir=None, prefixs=None):
        if prefixs is None:
            img_pths = glob(os.path.join(img_dir, '*.pkl'))

        else:
            img_pths = [os.path.join(img_dir, f'{prefix}.pkl') for prefix in prefixs]

        self.need_cluster = True if cluster_dir else False
        name_list, feat_list, omic_list, cluster_list = [], [], [], []
        for img_pth in img_pths:
            sample_name = img_pth.split('/')[-1].split('.')[0]
            omic_pth = os.path.join(omics_dir, f'{sample_name}.tsv')
            with open(img_pth, 'rb') as file:
                feat_dict = pkl.load(file)
            omic_df = pd.read_csv(omic_pth, sep='\t', index_col=0)
            for key, value in feat_dict.items():
                spot, spot_id = key.split('_')[-3], key.split('_')[-2]
                if f'{spot}_{spot_id}' not in omic_df.index:
                    print(f'{spot}_{spot_id}')
                    continue
                omic_feat = omic_df.loc[f'{spot}_{spot_id}']
                omic_list.append(np.array(omic_feat))
                feat_list.append(value.squeeze())
                name_list.append(key)
                if cluster_dir is not None:
                    cluster_df = pd.read_csv(os.path.join(cluster_dir, f'{sample_name}.tsv'), index_col=0, sep='\t')
                else:
                    continue
                cluster_feat = cluster_df.loc[f'{spot}_{spot_id}']
                cluster_list.append(cluster_feat['cluster'])
        self.name_list = name_list
        self.feat_list = feat_list
        self.omic_list = omic_list
        self.cluster_list = cluster_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        if self.need_cluster:
            neg_idx = generate_random_except(index, self.cluster_list, 0, len(self.feat_list) - 1)
        else:
            neg_idx = generate_random_except_without_cluster(index, 0, len(self.feat_list) - 1)
        return {
            'name': self.name_list[index],
            'image_feat': torch.tensor(self.feat_list[index].squeeze(), dtype=torch.float32).unsqueeze(0),
            'omic_feat': torch.tensor(self.omic_list[index], dtype=torch.float32).unsqueeze(0),
            'neg_feat': torch.tensor(self.feat_list[neg_idx].squeeze(), dtype=torch.float32).unsqueeze(0)
        }
    
class STImageDataset(Dataset):
    def __init__(self, img_dir, prefixs=None):
        if prefixs is None:
            img_pths = glob(os.path.join(img_dir, '*.pkl'))

        else:
            img_pths = [os.path.join(img_dir, f'{prefix}.pkl') for prefix in prefixs]

        name_list, feat_list = [], []
        for img_pth in img_pths:
            with open(img_pth, 'rb') as file:
                feat_dict = pkl.load(file)
            for key, value in feat_dict.items():
                feat_list.append(value.squeeze())
                name_list.append(key)
        self.name_list = name_list
        self.feat_list = feat_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        return {
            'name': self.name_list[index],
            'image_feat': torch.tensor(self.feat_list[index].squeeze(), dtype=torch.float32).unsqueeze(0),
        }

class VisiumDataset(Dataset):
    def __init__(self, img_pth, omics_pth, cluster_pth=None):
        name_list, feat_list, omic_list, cluster_list = [], [], [], []
        with open(img_pth, 'rb') as file:
            feat_dict = pkl.load(file)
        omic_df = pd.read_csv(omics_pth, sep='\t', index_col=0)
        # cluster_df = pd.read_csv(cluster_pth, index_col=0, sep='\t')
        for key, value in feat_dict.items():
            spot_id = key
            if spot_id not in omic_df.index:
                continue
            # if spot_id not in cluster_df.index:
            #     continue
            omic_feat = omic_df.loc[spot_id]
            omic_list.append(np.array(omic_feat))
            feat_list.append(value.squeeze())
            # cluster_list.append(cluster_df.loc[spot_id]['cluster'])
            name_list.append(key)

        self.name_list = name_list
        self.feat_list = feat_list
        self.omic_list = omic_list
        # self.cluster_list = cluster_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        neg_idx = generate_random_except_without_cluster(index, 0, len(self.feat_list) - 1)
        # neg_idx = generate_random_except(index, self.cluster_list, 0, len(self.feat_list) - 1)
        return {
            'name': self.name_list[index],
            'image_feat': torch.tensor(self.feat_list[index].squeeze(), dtype=torch.float32).unsqueeze(0),
            'omic_feat': torch.tensor(self.omic_list[index], dtype=torch.float32).unsqueeze(0),
            'neg_feat': torch.tensor(self.feat_list[neg_idx].squeeze(), dtype=torch.float32).unsqueeze(0)
        }

class BulkDataset(Dataset):
    def __init__(self, img_dir, omics_pth, lbl_pth=None, prefixs=None, max_img=10000, sample_gene=None):
        if prefixs is None:
            img_pths = glob(os.path.join(img_dir, '*.h5'))

        else:
            img_pths = [os.path.join(img_dir, f'{prefix}.h5') for prefix in prefixs]
        omics_df = pd.read_csv(omics_pth, sep='\t', index_col=0)
        # label_df = pd.read_csv(lbl_pth, sep='\t', index_col=0)
        name_list, feat_list, omic_list, cluster_list = [], [], [], []
        for img_pth in img_pths:
            # case_name = img_pth.split('/')[-1][:12]
            slide_name = img_pth.split('/')[-1].rstrip('.h5')
            sample_name = slide_name[:16]
            if sample_name not in omics_df.index:
                continue
            # if sample_name not in label_df.index:
            #     continue

            omic_df = omics_df.loc[[sample_name]]
            # cluster_df = label_df.loc[sample_name]
            # with h5py.File(img_pth, 'r') as hdf5_file:
            #     features = hdf5_file['features'][:]
                # coords = hdf5_file['coords'][:]
            name_list.append(slide_name)
            feat_list.append(img_pth)
            
            # cluster_list.append(cluster_df['cluster'])

            if sample_gene is not None:
                sub_df = omics_df.index.difference([sample_name])
                sampled_indices = np.random.choice(sub_df, size=sample_gene, replace=False)
                sampled_df = omics_df.loc[sampled_indices]
                omic_array = np.array(omic_df, dtype=np.float32)
                others_array = np.array(sampled_df, dtype=np.float32)
                all_omic_array = np.concatenate([omic_array, others_array], axis=0)
                omic_list.append(all_omic_array)    # N x G

            else:
                omic_list.append(np.array(omic_df, dtype=np.float32).squeeze())   # G

        self.name_list = name_list
        self.feat_list = feat_list
        self.omic_list = omic_list
        # self.cluster_list = cluster_list
        self.max_img = max_img

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        # neg_idx = generate_random_except(index, self.cluster_list, 0, len(self.feat_list) - 1)
        neg_idx = generate_random_except_without_cluster(index, 0, len(self.feat_list) - 1)
        img_pth = self.feat_list[index]
        with h5py.File(img_pth, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
        if features.shape[0] > self.max_img:
            np.random.shuffle(features)
            features = features[:self.max_img, :]

        neg_img_pth = self.feat_list[neg_idx]
        with h5py.File(neg_img_pth, 'r') as hdf5_file:
            neg_features = hdf5_file['features'][:]
        if neg_features.shape[0] > self.max_img:
            np.random.shuffle(neg_features)
            neg_features = neg_features[:self.max_img, :]

        return {
            'name': self.name_list[index],
            'image_feat': torch.tensor(features.squeeze(), dtype=torch.float32),
            'omic_feat': torch.tensor(self.omic_list[index], dtype=torch.float32).unsqueeze(0),
            'neg_feat': torch.tensor(neg_features.squeeze(), dtype=torch.float32)
        }
        