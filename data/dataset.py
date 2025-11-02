import os
import numpy as np
import pickle as pkl
import pandas as pd
import random
from scipy import cluster
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
    def __init__(self, img_dir, omics_dir, prefixs=None, index_type='barcode'):
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
                if index_type in ['barcode']:
                    index = key.split('_')[0]

                elif index_type in ['xy']:
                    index = key.split('_')[-2]

                elif index_type in ['file_name']:
                    index = key

                else:
                    index = key.split('_')[1]
                
                if index not in omic_df.index:
                    print(index)
                    continue
                omic_feat = omic_df.loc[index]
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
    
class XeniumLungDataset(Dataset):
    def __init__(self, img_pth, omic_pth):
        with open(img_pth, 'rb') as file:
            feat_dict = pkl.load(file)

        omic_df = pd.read_csv(omic_pth, sep='\t', index_col=0)
        name_list, feat_list, omic_list = [], [], []
        for key, value in feat_dict.items():
            spot_id = int(key.split('_')[0])
            if f'spot_{spot_id}' not in omic_df.index:
                print(f'spot_{spot_id}')
                continue
            omic_feat = omic_df.loc[f'spot_{spot_id}']
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
    
class STImageDataset(Dataset):
    def __init__(self, img_dir, prefixs=None, ext='pkl'):
        if prefixs is None:
            img_pths = glob(os.path.join(img_dir, f'*.{ext}'))

        else:
            img_pths = [os.path.join(img_dir, f'{prefix}.{ext}') for prefix in prefixs]

        name_list, feat_list = [], []
        for img_pth in img_pths:
            if ext in ['pkl']:
                with open(img_pth, 'rb') as file:
                    feat_dict = pkl.load(file)
                for key, value in feat_dict.items():
                    feat_list.append(value.squeeze())
                    name_list.append(key)

            elif ext in ['h5']:
                slide_name = img_pth.split('/')[-1].split('.h5')[0]
                with h5py.File(img_pth, 'r') as hdf5_file:
                    features = hdf5_file['features'][:]
                    coords = hdf5_file['coords'][:]

                for feature, coord in zip(features, coords):
                    feat_list.append(feature.squeeze())
                    name_list.append(f"{slide_name}_{coord[0]}x{coord[1]}")

            else:
                raise NotImplementedError
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
    def __init__(self, img_pth, omics_pth):
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
    def __init__(self, img_dir, omics_pth, prefixs=None, max_img=10000, sample_gene=None):
        if prefixs is None:
            img_pths = glob(os.path.join(img_dir, '*.h5'))
            train_number = len(img_pths)

        else:
            train_indices = []
            for prefix in prefixs:
                train_indices.extend(glob(os.path.join(img_dir, f'{prefix}*.h5')))
            train_number = len(train_indices)
            img_pths = train_indices.copy()
            for img_pth in glob(os.path.join(img_dir, '*.h5')):
                if img_pth not in img_pths:
                    img_pths.append(img_pth)

        omics_df = pd.read_csv(omics_pth, sep='\t', index_col=0)
        
        name_list, feat_list, omic_list, cluster_list = [], [], [], []
        for img_pth in img_pths:
            if not os.path.exists(img_pth):
                print(img_pth)
                continue
            slide_name = img_pth.split('/')[-1].rstrip('.h5')+'.svs'
            sample_name = slide_name[:16]
            if sample_name not in omics_df.index:
                continue
            omic_df = omics_df.loc[[sample_name]]
            name_list.append(slide_name)
            feat_list.append(img_pth)
            if sample_gene is not None:
                sub_df = omics_df.index.difference([slide_name])
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
        self.max_img = max_img

    def __len__(self):
        return len(self.omic_list)
    
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

class BulkImageDataset(Dataset):
    def __init__(self, img_dir, prefixs=None, max_img=10000):
        if prefixs is None:
            img_pths = glob(os.path.join(img_dir, '*.h5'))

        else:
            img_pths = []
            for prefix in prefixs:
                img_pths.extend(glob(os.path.join(img_dir, f'{prefix}*.h5')))

        name_list, feat_list = [], []
        for img_pth in img_pths:
            slide_name = img_pth.split('/')[-1].rstrip('.h5')+'.svs'
            name_list.append(slide_name)
            feat_list.append(img_pth)

        self.name_list = name_list
        self.feat_list = feat_list
        self.max_img = max_img

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        img_pth = self.feat_list[index]
        with h5py.File(img_pth, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
        # if features.shape[0] > self.max_img:
        #     np.random.shuffle(features)
        #     features = features[:self.max_img, :]

        return {
            'name': self.name_list[index],
            'image_feat': torch.tensor(features.squeeze(), dtype=torch.float32)
        }


class MutationDataset(Dataset):
    def __init__(self, img_dir, omics_pth, prefixs=None, max_img=10000, sample_gene=None):
        if prefixs is None:
            img_pths = glob(os.path.join(img_dir, '*.h5'))

        else:
            img_pths = []
            for prefix in prefixs:
                slide_name = prefix.split('.svs')[0]
                img_pths.append(os.path.join(img_dir, f'{slide_name}.h5'))
        omics_df = pd.read_csv(omics_pth, sep='\t', index_col=0)
        name_list, feat_list, omic_list, cluster_list = [], [], [], []
        for img_pth in img_pths:
            # case_name = img_pth.split('/')[-1][:12]
            slide_name = img_pth.split('/')[-1].rstrip('.h5') + '.svs'
            if slide_name not in omics_df.index:
                continue
            # if sample_name not in label_df.index:
            #     continue

            omic_df = omics_df.loc[[slide_name]]
            # cluster_df = label_df.loc[sample_name]
            # with h5py.File(img_pth, 'r') as hdf5_file:
            #     features = hdf5_file['features'][:]
                # coords = hdf5_file['coords'][:]
            name_list.append(slide_name)
            feat_list.append(img_pth)
            
            # cluster_list.append(cluster_df['cluster'])

            if sample_gene is not None:
                sub_df = omics_df.index.difference([slide_name])
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


class MultiOmicsDataset(Dataset):
    def __init__(self, omic_a_pth, omic_b_pth):
        name_list, feat_list, omic_list = [], [], []
        omic_a_df = pd.read_csv(omic_a_pth, sep='\t', index_col=0)
        omic_b_df = pd.read_csv(omic_b_pth, sep='\t', index_col=0)
        for spot in omic_a_df.index:
            if spot not in omic_b_df.index:
                print(f"Warning! {spot} not in omics b sequenced list.")
            feat_list.append(np.array(omic_a_df.loc[spot]))
            omic_list.append(np.array(omic_b_df.loc[spot]))
            name_list.append(spot)

        self.name_list = name_list
        self.feat_list = feat_list
        self.omic_list = omic_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        neg_idx = generate_random_except_without_cluster(index, 0, len(self.feat_list) - 1)
        # neg_idx = generate_random_except(index, self.cluster_list, 0, len(self.feat_list) - 1)
        return {
            'name': self.name_list[index],
            'protein_feat': torch.tensor(self.feat_list[index].squeeze(), dtype=torch.float32).unsqueeze(0),
            'omic_feat': torch.tensor(self.omic_list[index], dtype=torch.float32).unsqueeze(0),
            'neg_feat': torch.tensor(self.feat_list[neg_idx].squeeze(), dtype=torch.float32).unsqueeze(0)
        }
    

class BulkRNAProteinDataset(Dataset):
    def __init__(self, img_dir, rna_pth, protein_pth, prefixs=None, max_img=10000, sample_gene=None):
        if prefixs is None:
            img_pths = glob(os.path.join(img_dir, '*.h5'))

        else:
            train_indices = [os.path.join(img_dir, f'{prefix}.h5') for prefix in prefixs]
            img_pths = train_indices.copy()
            for img_pth in glob(os.path.join(img_dir, '*.h5')):
                if img_pth not in img_pths:
                    img_pths.append(img_pth)

        rna_df = pd.read_csv(rna_pth, sep='\t', index_col=0)
        pro_df = pd.read_csv(protein_pth, sep='\t', index_col=0)

        name_list, feat_list, rna_list, protein_list = [], [], [], []
        other_feat_list, other_rna_list, other_protein_list = [], [], []
        sample_name_list = []
        paired_numbers = 0
        for img_pth in img_pths:
            if not os.path.exists(img_pth):
                other_feat_list.append(img_pth)
                continue
            slide_name = img_pth.split('/')[-1].rstrip('.h5')
            sample_name = slide_name[:16]
            if sample_name not in rna_df.index:
                # other_rna_list.append(np.array(rna_single_df, dtype=np.float32).squeeze())
                continue
            if sample_name not in pro_df.index:
                # other_protein_list.append(np.array(pro_single_df, dtype=np.float32).squeeze())
                continue

            # Paired samples
            paired_numbers += 1
            rna_single_df = rna_df.loc[[sample_name]]
            pro_single_df = pro_df.loc[[sample_name]]

            name_list.append(slide_name)
            sample_name_list.append(sample_name)
            feat_list.append(img_pth)

            rna_list.append(np.array(rna_single_df, dtype=np.float32).squeeze())   # G
            protein_list.append(np.array(pro_single_df, dtype=np.float32).squeeze())   # G

        self.name_list = name_list
        feat_list.extend(other_feat_list)
        self.feat_list = feat_list
        
        for index, single_df in rna_df.iterrows():
            if index not in sample_name_list:
                other_rna_list.append(np.array(single_df, dtype=np.float32).squeeze())

        for index, single_df in pro_df.iterrows():
            if index not in sample_name_list:
                other_protein_list.append(np.array(single_df, dtype=np.float32).squeeze())
        
        rna_list.extend(other_rna_list)
        self.rna_list = rna_list

        protein_list.extend(other_protein_list)
        self.protein_list = protein_list

        self.max_img = max_img
        self.paired_numbers = paired_numbers
        # import ipdb; ipdb.set_trace()

    def __len__(self):
        return self.paired_numbers
    
    def __getitem__(self, index):
        # Sampling pathology image
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

        neg_idx = generate_random_except_without_cluster(index, 0, len(self.rna_list) - 1)
        rna_expr = self.rna_list[index]
        neg_rna_expr = self.rna_list[neg_idx]

        neg_idx = generate_random_except_without_cluster(index, 0, len(self.protein_list) - 1)
        pro_expr = self.protein_list[index]
        neg_pro_expr = self.protein_list[neg_idx]
        
        return {
            'name': self.name_list[index],
            'img': torch.tensor(features.squeeze(), dtype=torch.float32),
            'neg_img': torch.tensor(neg_features.squeeze(), dtype=torch.float32),
            'rna': torch.tensor(rna_expr, dtype=torch.float32).unsqueeze(0),
            'neg_rna': torch.tensor(neg_rna_expr, dtype=torch.float32).unsqueeze(0),
            'pro': torch.tensor(pro_expr.squeeze(), dtype=torch.float32).unsqueeze(0),
            'neg_pro': torch.tensor(neg_pro_expr.squeeze(), dtype=torch.float32).unsqueeze(0)
        }
    

class STRNAProteinDataset(Dataset):
    def __init__(self, img_pth, rna_pth, protein_pth):
        name_list, feat_list, rna_list, protein_list = [], [], [], []
        rna_df = pd.read_csv(rna_pth, sep='\t', index_col=0)
        protein_df = pd.read_csv(protein_pth, sep='\t', index_col=0)

        with open(img_pth, 'rb') as file:
            feat_dict = pkl.load(file)

        for spot_xy, feat in feat_dict.items():
            barcode = spot_xy.split('_')[0]
            if barcode not in rna_df.index:
                continue
            if barcode not in protein_df.index:
                continue

            name_list.append(spot_xy)
            feat_list.append(feat)
            rna_list.append(np.array(rna_df.loc[barcode]))
            protein_list.append(np.array(protein_df.loc[barcode]))

        self.name_list = name_list
        self.feat_list = feat_list
        self.rna_list = rna_list
        self.protein_list = protein_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        # Sampling pathology image
        neg_idx = generate_random_except_without_cluster(index, 0, len(self.feat_list) - 1)
        feature = self.feat_list[index]
        neg_feature = self.feat_list[neg_idx]

        neg_idx = generate_random_except_without_cluster(index, 0, len(self.rna_list) - 1)
        rna_expr = self.rna_list[index]
        neg_rna_expr = self.rna_list[neg_idx]

        neg_idx = generate_random_except_without_cluster(index, 0, len(self.protein_list) - 1)
        pro_expr = self.protein_list[index]
        neg_pro_expr = self.protein_list[neg_idx]
        
        return {
            'name': self.name_list[index],
            'img': torch.tensor(feature, dtype=torch.float32).unsqueeze(0),
            'neg_img': torch.tensor(neg_feature, dtype=torch.float32).unsqueeze(0),
            'rna': torch.tensor(rna_expr, dtype=torch.float32).unsqueeze(0),
            'neg_rna': torch.tensor(neg_rna_expr, dtype=torch.float32).unsqueeze(0),
            'pro': torch.tensor(pro_expr, dtype=torch.float32).unsqueeze(0),
            'neg_pro': torch.tensor(neg_pro_expr, dtype=torch.float32).unsqueeze(0)
        }