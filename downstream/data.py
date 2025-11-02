from builtins import NotImplemented, NotImplementedError
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import os
import h5py
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm

class RNADataset(Dataset):
    def __init__(self, img_pth, omics_pth, prefixs=None, datatype='RNA', sampling=False):
        self.datatype = datatype
        with open(img_pth, 'rb') as file:
            feat_dict = pkl.load(file)

        omics_df = pd.read_csv(omics_pth, sep='\t', index_col=0)
        self.gene_list = omics_df.keys().to_list()
        self.omics_dim = omics_df.shape[1]
        
        if prefixs is not None:
            data_names = prefixs
        else:
            data_names = feat_dict['names']

        omics_list, feat_list = [], []
        name_list = []

        for name, feat in zip(feat_dict['names'], feat_dict['images']):
            name = name.split('.svs')[0]
        # for name in data_names:
            if datatype in ['RNA']:
                if name.startswith('TCGA'):
                    sample_name = name[:16]

                else:
                    sample_name = name.split('.')[0]
            else:
                sample_name = name
            if prefixs is not None and sample_name not in data_names and name not in data_names:
                continue

            if sample_name not in omics_df.index:
                continue

            omics_list.append(np.array(omics_df.loc[sample_name]))
            feat_list.append(feat)
            name_list.append(name)

        self.name_list = name_list
        self.omics_list = omics_list
        self.feat_list = feat_list
        self.sampling = sampling

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        name = self.name_list[index]

        # omic_dtype = torch.float32 if self.datatype in ['RNA'] else torch.long
        omic_feat = torch.tensor(self.omics_list[index], dtype=torch.float32).squeeze()
        img_feat = torch.tensor(self.feat_list[index], dtype=torch.float32).squeeze()
        if self.sampling and img_feat.shape[0] > 10000:
            sampled_features = np.random.choice(img_feat.shape[0], 10000, replace=False)
            img_feat = img_feat[sampled_features].squeeze()
        return {
            'name': name,
            'omic_feat': omic_feat,
            'img_feat': img_feat
        }
    
class PTData(Dataset):
    def __init__(self, img_pth, omics_pth, prefixs=None):
        feat_array = np.load(img_pth, allow_pickle=True)
        omics_df = pd.read_csv(omics_pth, sep='\t', index_col=0)
        
        if prefixs is not None:
            data_names = [prefix[:16] for prefix in prefixs]
        else:
            data_names = [item[0].split('_')[-1][:16] for item in feat_array]

        omics_list, feat_list = [], []
        name_list = []
        for feat in feat_array:
            name, img_feat = feat[0], feat[1]
            sample_name = name.split('_')[-1][:16]
            if sample_name not in data_names:
                print(f"{sample_name} not in list.")
                continue
            name_list.append(sample_name)
            omics_list.append(np.array(omics_df.loc[sample_name]))
            feat_list.append(img_feat)
        self.name_list = name_list
        self.omics_list = omics_list
        self.feat_list = feat_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        name = self.name_list[index]
        omic_feat = torch.tensor(self.omics_list[index], dtype=torch.float32).squeeze()
        img_feat = torch.tensor(self.feat_list[index], dtype=torch.float32).squeeze()
        
        if img_feat.shape[0] > 5000:
            sampled_features = np.random.choice(img_feat.shape[0], 5000, replace=False)
            img_feat = img_feat[sampled_features]
        return {
            'name': name,
            'omic_feat': omic_feat,
            'img_feat': img_feat
        }

class MultiSTImageDataset(RNADataset):
    def __init__(self, img_dir, omic_dir, prefixs=None, exp='Multi-Embed'):
        if prefixs is not None:
            data_names = prefixs
        else:
            data_names = [item[0].split('.')[0] for item in os.listdir(img_dir)]

        exp_list, feat_list, name_list = [], [], []

        for prefix in data_names:
            img_pth = os.path.join(img_dir, f"{prefix}.pkl")

            with open(img_pth, 'rb') as file:
                feat_dict = pkl.load(file)

            omics_df = pd.read_csv(os.path.join(omic_dir, f"{prefix}.tsv"), sep='\t', index_col=0)

            if exp in ['Multi-Embed']:
                if feat_dict.get('names') is None:
                    iterations = zip(feat_dict.keys(), feat_dict.values())
                else:
                    # new_name_list = [item.split('_')[-2] for item in feat_dict['names']]
                    iterations = zip(feat_dict['names'], feat_dict['agg_images'])
            
            elif exp in ['iStar']:
                iterations = feat_dict.items()

            elif exp in ['OmiCLIP']:
                iterations = feat_dict.items()

            else:
                raise NotImplementedError
            
            # import ipdb; ipdb.set_trace
            for spot_name, feat_vec in iterations:
                if not prefix.startswith('lung'):
                    spot_name = spot_name.split('_')[-2]
                if prefix.startswith('H1'):
                    x, y = spot_name.split('x')
                    spot_name = f"{y}x{x}"

                if spot_name not in omics_df.index:
                    # import ipdb; ipdb.set_trace()
                    continue
                exp_list.append(np.array(omics_df.loc[spot_name]))
                feat_list.append(feat_vec)
                name_list.append(spot_name)

        self.name_list = name_list
        self.omics_list = exp_list
        self.feat_list = feat_list
        self.sampling = False


class VisiumHDGeneDataset(RNADataset):
    def __init__(self, img_pth, omic_pth, exp='Multi-Embed'):
        exp_list, feat_list, name_list = [], [], []
        with open(img_pth, 'rb') as file:
            feat_dict = pkl.load(file)
        omics_df = pd.read_csv(omic_pth, sep='\t', index_col=0)

        if exp in ['Multi-Embed']:
            # import ipdb; ipdb.set_trace()
            # feat_mat = np.concatenate(feat_dict['images'], axis=0)
            # XXX(huangxuan): Temp
            if feat_dict.get('names') is None:
                iterations = zip(feat_dict.keys(), feat_dict.values())
            else:
                iterations = zip(feat_dict['names'], feat_dict['agg_images'])
        
        elif exp in ['iStar']:
            iterations = feat_dict.items()

        elif exp in ['OmiCLIP']:
            iterations = feat_dict.items()

        else:
            raise NotImplementedError
        
        for spot_name, feat_vec in iterations:
            if spot_name not in omics_df.index:
                continue
            exp_list.append(np.array(omics_df.loc[spot_name]))
            feat_list.append(feat_vec)
            name_list.append(spot_name)

        self.name_list = name_list
        self.omics_list = exp_list
        self.feat_list = feat_list
        self.sampling = False

class STGeneDataset(RNADataset):
    def __init__(self, img_dir, omic_dir, prefixs=None):
        omics_list, feats_list, name_list = [], [], []
        for name in prefixs:
            with open(os.path.join(img_dir, f'{name}_results.pkl'), 'rb') as file:
                feat_dict = pkl.load(file)
            omics_df = pd.read_csv(os.path.join(omic_dir, f'{name}.tsv'), sep='\t', index_col=0)
            feat_mat = np.concatenate(feat_dict['images'], axis=0)
            exp_list, feat_list = [], []
            for spot_name, feat_vec in zip(feat_dict['names'], feat_mat):
                spot, spot_id = spot_name.split('_')[-3], spot_name.split('_')[-2]
                if spot in ['spot']:
                    spot_id = f"{spot}_{spot_id}"
                if spot_id not in omics_df.index:
                    continue
                exp_list.append(omics_df.loc[spot_id])
                feat_list.append(feat_vec)

            cnts = np.stack(exp_list)
            cnts_min = cnts.min(0)
            cnts_max = cnts.max(0)
            cnts -= cnts_min
            ncnts = cnts / ((cnts_max - cnts_min) + 1e-12)
            filtered_feat_mat = np.stack(feat_list)
            omics_list.append(ncnts)
            name_list.append(name)
            feats_list.append(filtered_feat_mat)

        self.name_list = name_list
        self.omics_list = omics_list
        self.feat_list = feats_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        name = self.name_list[index]
        omic_feat = torch.tensor(self.omics_list[index], dtype=torch.float32).squeeze()
        img_feat = torch.tensor(self.feat_list[index], dtype=torch.float32).squeeze()
        return {
            'name': name,
            'omic_feat': omic_feat,
            'img_feat': img_feat
        }
    
class XeniumGeneDataset(RNADataset):
    def __init__(self, img_dir, omic_dir, prefixs=None):
        omics_list, feats_list, name_list = [], [], []
        for name in prefixs:
            with open(os.path.join(img_dir, f'{name}.pkl'), 'rb') as file:
                feat_dict = pkl.load(file)
            omics_df = pd.read_csv(os.path.join(omic_dir, f'{name}.tsv'), sep='\t', index_col=0)
            
            exp_list, feat_list = [], []

            if 'names' in feat_dict.keys():
                for spot_name, feat_vec in zip(feat_dict['names'], np.concatenate(feat_dict['images'])):
                    spot_id = int(spot_name.split('_')[0])
                    exp_list.append(omics_df.loc[f"spot_{spot_id}"])
                    feat_list.append(feat_vec)

            else:
                for spot_name, feat_vec in feat_dict.items():
                    spot_id = int(spot_name.split('_')[0])
                    exp_list.append(omics_df.loc[f"spot_{spot_id}"])
                    feat_list.append(feat_vec)
                    # spot, spot_id = spot_name.split('_')[-3], spot_name.split('_')[-2]
                    # if spot in ['spot']:
                    #     spot_id = f"{spot}_{spot_id}"
                    # if spot_id not in omics_df.index:
                    #     continue
                    # exp_list.append(omics_df.loc[spot_id])
                    # feat_list.append(feat_vec)

            cnts = np.stack(exp_list)
            cnts_min = cnts.min(0)
            cnts_max = cnts.max(0)
            cnts -= cnts_min
            ncnts = cnts / ((cnts_max - cnts_min) + 1e-12)
            filtered_feat_mat = np.stack(feat_list)
            omics_list.append(ncnts)
            name_list.append(name)
            feats_list.append(filtered_feat_mat)

        self.name_list = name_list
        self.omics_list = omics_list
        self.feat_list = feats_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        name = self.name_list[index]
        omic_feat = torch.tensor(self.omics_list[index], dtype=torch.float32).squeeze()
        img_feat = torch.tensor(self.feat_list[index], dtype=torch.float32).squeeze()
        return {
            'name': name,
            'omic_feat': omic_feat,
            'img_feat': img_feat
        }
    

class STRawGeneDataset(RNADataset):
    def __init__(self, img_dir, omic_dir, prefixs=None):
        omics_list, feats_list, name_list = [], [], []
        for name in prefixs:
            with open(os.path.join(img_dir, f'{name}.pkl'), 'rb') as file:
                feat_dict = pkl.load(file)
            omics_df = pd.read_csv(os.path.join(omic_dir, f'{name}.tsv'), sep='\t', index_col=0)
            exp_list, feat_list = [], []
            # import ipdb; ipdb.set_trace()
            for spot_name, feat_vec in feat_dict.items():
                xy = spot_name.split('_')[1]
                # xy = spot_name
                if xy not in omics_df.index:
                    continue
                exp_list.append(omics_df.loc[xy])
                feat_list.append(feat_vec)

            exp_mat = np.stack(exp_list)
            filtered_feat_mat = np.stack(feat_list)
            omics_list.append(exp_mat)
            name_list.append(name)
            feats_list.append(filtered_feat_mat)

        self.name_list = name_list
        self.omics_list = omics_list
        self.feat_list = feats_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        name = self.name_list[index]
        omic_feat = torch.tensor(self.omics_list[index], dtype=torch.float32).squeeze()
        img_feat = torch.tensor(self.feat_list[index], dtype=torch.float32).squeeze()
        return {
            'name': name,
            'omic_feat': omic_feat,
            'img_feat': img_feat
        }

    
class ImageFolder(Dataset):
    def __init__(self, img_dir, omics_pth, prefixs=None, sampling=False, ext='h5', datatype='RNA', transpose=True) -> None:
        super().__init__()
        self.sampling = sampling
        self.ext = ext
        self.transpose = transpose
        if prefixs is None:
            img_pths = glob(os.path.join(img_dir, f'*.{ext}'))

        else:
            if datatype in ['RNA']:
                img_pths = [os.path.join(img_dir, f'{prefix}.{ext}') for prefix in prefixs]
            else:
                img_pths = []
                for prefix in prefixs:
                    if prefix.endswith('.svs'):
                        img_pths.append(os.path.join(img_dir, prefix.replace('.svs', f'.{ext}')))
                    else:
                        img_pths.extend(glob(os.path.join(img_dir, f"{prefix}*.{ext}")))

        omics_df = pd.read_csv(omics_pth, sep='\t', index_col=0)
        name_list, feat_list, omic_list = [], [], []
        # import ipdb; ipdb.set_trace()
        for img_pth in img_pths:
            if not os.path.exists(img_pth):
                continue
            if datatype in ['RNA', 'Meth', 'Protein']:
                sample_name = img_pth.split('/')[-1].split('.')[0][:16]
            elif datatype in ['CPTAC-2']:
                sample_name = img_pth.split('/')[-1].split('.')[0]
            else:
                sample_name = img_pth.split('/')[-1].split('.h5')[0] + '.svs'
            if sample_name not in omics_df.index:
                continue
            omic_df = omics_df.loc[sample_name]

            name_list.append(sample_name)
            feat_list.append(img_pth)
            omic_list.append(np.array(omic_df, dtype=np.float32))

        self.name_list = name_list
        self.feat_list = feat_list
        self.omic_list = omic_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        name = self.name_list[index]
        img_pth = self.feat_list[index]
        if self.ext in ['h5']:
            with h5py.File(img_pth, 'r') as hdf5_file:
                features = hdf5_file['features'][:]
            # img_feat = torch.tensor(features, dtype=torch.float32).squeeze().transpose(-1, -2)
            # print(img_feat.shape)
        else:
            features = np.load(img_pth)
        img_feat = torch.tensor(features.squeeze(), dtype=torch.float32)
        
        if self.sampling and img_feat.shape[0] > 10000:
            sampled_features = np.random.choice(img_feat.shape[0], 10000, replace=False)
            img_feat = img_feat[sampled_features].squeeze()
        omic_feat = torch.tensor(self.omic_list[index], dtype=torch.float32).squeeze()

        if self.transpose:
            img_feat = img_feat.transpose(-1, -2)
        return {
            'name': name,
            'omic_feat': omic_feat,
            'img_feat': img_feat
        }
    
def convert_time_to_period(time, cutoffs=[]):
    if time > cutoffs[3]:
        return 3
    elif time > cutoffs[2]:
        return 2
    elif time > cutoffs[1]:
        return 1
    else:
        return 0

class FusionDataset(Dataset):
    def __init__(self, feat_pth, prefixs=None, survival_pth=''):
        with open(feat_pth, 'rb') as file:
            feat_dict = pkl.load(file)
        
        if prefixs is not None:
            data_names = prefixs
        else:
            data_names = feat_dict['names']

        omics_list, feat_list = [], []
        agg_list = []
        rna_list, protein_list = [], []
            
        surv_df = pd.read_csv(survival_pth, index_col=0)
        surv_df = surv_df[surv_df['vital_status'] != '[Not Available]']
        sub_df = surv_df[surv_df['vital_status'] == 'Dead']
        t0 = 0
        death_days = np.array(sub_df['death_days_to'], dtype=np.int32)
        t1, t2, t3 = np.quantile(death_days, q=0.25), np.median(death_days), np.quantile(death_days, q=0.75)
        censor_list, time_list, hazard_list = [], [], []
        
        self.multiomics = 'rna' in feat_dict.keys() and 'protein' in feat_dict.keys()
        for prefix in data_names:
            if prefix.startswith('TCGA'):
                sample_id = prefix[:12]
            else:
                sample_id = prefix

            sample_name = prefix if prefix.endswith('.svs') else prefix + '.svs' 
            if sample_id not in surv_df.index:
                continue
            if sample_name not in feat_dict['names']:
                continue
            
            name_idx = feat_dict['names'].index(sample_name)
            if not self.multiomics:
                omics_list.append(feat_dict['omics'][name_idx])
            else:
                # omics_list.append(np.concatenate([feat_dict['rna'][name_idx], feat_dict['protein'][name_idx]]))
                # omics_list.append(feat_dict['rna'][name_idx])
                omics_list.append(np.stack([feat_dict['rna'][name_idx], feat_dict['protein'][name_idx]]))
            
            agg_list.append(feat_dict['agg_images'][name_idx])
            feat_list.append(feat_dict['images'][name_idx])
            single_df = surv_df.loc[[sample_id]].iloc[0]
            censor_list.append(single_df['vital_status'] == 'Alive')
            surv_time = single_df['last_contact_days_to'] + single_df['death_days_to'] + 1
            time_list.append(surv_time)
            hazard_list.append(convert_time_to_period(surv_time, [t0, t1, t2, t3]))

        self.name_list = data_names
        self.feat_list = feat_list
        self.agg_list = agg_list

        if not self.multiomics:
            self.omic_list = omics_list
        else:
            omics_array = np.stack(omics_list)
            self.omic1_list = omics_array[:, 0]
            self.omic2_list = omics_array[:, 1]

        self.censor_list = censor_list
        self.time_list = time_list
        self.hazard_list = hazard_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        item_dict = {}
        name = self.name_list[index]
        img_feat = torch.tensor(self.feat_list[index], dtype=torch.float32).squeeze()
        if self.multiomics:
            omic1_feat = torch.tensor(self.omic1_list[index], dtype=torch.float32).squeeze()
            omic2_feat = torch.tensor(self.omic2_list[index], dtype=torch.float32).squeeze()
            item_dict.update(
                {
                    'omic1_feat': omic1_feat,
                    'omic2_feat': omic2_feat
                }
            )

        else:
            omic_feat = torch.tensor(self.omic_list[index], dtype=torch.float32).squeeze()
            item_dict.update(
                {
                    'omic_feat': omic_feat
                }
            )

        agg_feat = torch.tensor(self.agg_list[index], dtype=torch.float32).squeeze()
        censor = torch.tensor(self.censor_list[index], dtype=torch.float32)
        hazard = torch.tensor(self.hazard_list[index], dtype=torch.long)
        time = torch.tensor(self.time_list[index], dtype=torch.float32) / 30
        item_dict.update(
                {
                'name': name,
                'agg_feat': agg_feat,
                'img_feat': img_feat,
                'censor': censor,
                'time': hazard,
                'months': time
            }
        )
        return item_dict

class GeneRawDataset(FusionDataset):
    def __init__(self, omics_pth: str, prefixs=None, survival_pth=''):
        if omics_pth.endswith('csv.zip'):
            omic_df = pd.read_csv(omics_pth, index_col=0)
            del omic_df['slide_id']
            del omic_df['train']
            del omic_df['site']
            del omic_df['is_female']
            del omic_df['oncotree_code']
            del omic_df['age']
            del omic_df['survival_months']
            del omic_df['censorship']
        else:
            omic_df = pd.read_csv(omics_pth, sep='\t', index_col=0)

        omics_list, feat_list = [], []
            
        surv_df = pd.read_csv(survival_pth, index_col=0)
        surv_df = surv_df[surv_df['vital_status'] != '[Not Available]']
        sub_df = surv_df[surv_df['vital_status'] == 'Dead']
        t0 = 0
        death_days = np.array(sub_df['death_days_to'], dtype=np.int32)
        t1, t2, t3 = np.quantile(death_days, q=0.25), np.median(death_days), np.quantile(death_days, q=0.75)
        censor_list, time_list, hazard_list, name_list = [], [], [], []
        for prefix in prefixs:
            sample_id = prefix[:12]
            sample_name = prefix[:16]
            if sample_id not in surv_df.index:
                continue
            if sample_id not in omic_df.index:
                continue
            name_list.append(sample_name)
            omics_list.append(np.array(omic_df.loc[[sample_id]].iloc[0]))
            single_df = surv_df.loc[[sample_id]].iloc[0]
            censor_list.append(single_df['vital_status'] == 'Alive')
            surv_time = single_df['last_contact_days_to'] + single_df['death_days_to'] + 1
            time_list.append(surv_time)
            hazard_list.append(convert_time_to_period(surv_time, [t0, t1, t2, t3]))

        self.name_list = name_list
        self.feat_list = omics_list
        self.omic_list = omics_list
        self.censor_list = censor_list
        self.time_list = time_list
        self.hazard_list = hazard_list


class FusionRawDataset(Dataset):
    def __init__(self, img_dir, omics_pth, prefixs=None, survival_pth='') -> None:
        super().__init__()
        if prefixs is None:
            img_pths = glob(os.path.join(img_dir, '*.h5'))

        else:
            img_pths = [os.path.join(img_dir, f'{prefix}.h5') for prefix in prefixs]
        omic_df = pd.read_csv(omics_pth, index_col=0)
        del omic_df['slide_id']
        del omic_df['train']
        del omic_df['site']
        del omic_df['is_female']
        del omic_df['oncotree_code']
        del omic_df['age']
        del omic_df['survival_months']
        del omic_df['censorship']

        surv_df = pd.read_csv(survival_pth, index_col=0)
        surv_df = surv_df[surv_df['vital_status'] != '[Not Available]']
        sub_df = surv_df[surv_df['vital_status'] == 'Dead']
        t0 = 0
        death_days = np.array(sub_df['death_days_to'], dtype=np.int32)
        t1, t2, t3 = np.quantile(death_days, q=0.25), np.median(death_days), np.quantile(death_days, q=0.75)

        name_list, omics_list = [], []
        censor_list, time_list, hazard_list = [], [], []

        for prefix in prefixs:
            sample_id = prefix[:12]
            sample_name = prefix[:16]
            if sample_id not in surv_df.index:
                continue
            if sample_id not in omic_df.index:
                continue
            name_list.append(sample_name)
            omics_list.append(np.array(omic_df.loc[[sample_id]].iloc[0]))
            single_df = surv_df.loc[[sample_id]].iloc[0]
            censor_list.append(single_df['vital_status'] == 'Alive')
            surv_time = single_df['last_contact_days_to'] + single_df['death_days_to'] + 1
            time_list.append(surv_time)
            hazard_list.append(convert_time_to_period(surv_time, [t0, t1, t2, t3]))

        self.name_list = name_list
        self.feat_list = img_pths
        self.omic_list = omics_list
        self.censor_list = censor_list
        self.time_list = time_list
        self.hazard_list = hazard_list

    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, index):
        name = self.name_list[index]
        img_pth = self.feat_list[index]
        with h5py.File(img_pth, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
        img_feat = torch.tensor(features, dtype=torch.float32).squeeze()
        omic_feat = torch.tensor(self.omic_list[index], dtype=torch.float32).squeeze()
        censor = torch.tensor(self.censor_list[index], dtype=torch.float32)
        hazard = torch.tensor(self.hazard_list[index], dtype=torch.long)
        time = torch.tensor(self.time_list[index], dtype=torch.float32) / 30
        return {
            'name': name,
            'omic_feat': omic_feat,
            'img_feat': img_feat,
            'censor': censor,
            'time': hazard,
            'months': time
        }
    

class SurvivalDataset(Dataset):
    def __init__(self, feat_dir, omic_dir=None, survival_pth='', mode='TNBC'):    
        surv_df = pd.read_csv(survival_pth, index_col=0)
        t0 = 0
        death_days = np.array(surv_df['death_days_to'].dropna(), dtype=np.int32)
        t1, t2, t3 = np.quantile(death_days, q=0.25), np.median(death_days), np.quantile(death_days, q=0.75)
        censor_list, time_list, hazard_list = [], [], []

        data_names = []
        omics_list, feat_list = [], []
        for feat_pth in glob(os.path.join(feat_dir, '*.pkl')):
            if mode in ['TNBC']:
                slide_name = feat_pth.split('/')[-2]
                slide_file = glob(f'/data/gcf22/Spatial/BRCA/TNBC_ST/Images/imagesHD/*_{slide_name}.jpg')[0]
                slide_id = int(slide_file.split('/')[-1].split('_')[0].lstrip('TNBC'))

            else:
                slide_name = feat_pth.split('/')[-1].split('_results.pkl')[0]
                slide_id = slide_name

            if slide_id not in surv_df.index:
                print(f"{slide_id} not in clinical dataframe.")
                continue

            if pd.isna(surv_df.loc[slide_id]['vital_status']):
                continue

            with open(feat_pth, 'rb') as file:
                feat_dict = pkl.load(file)
            
            if 'agg_images' not in feat_dict.keys():
                img_feats = np.stack(list(feat_dict.values()))
            else:
                img_feats = feat_dict['agg_images']

            if omic_dir is None:
                omics_list.append(np.zeros_like(img_feats))
            else:
                pass

            data_names.append(slide_name)
            feat_list.append(img_feats)
            single_df = surv_df.loc[slide_id]
            if mode in ['TNBC']:
                Alive_symbol = 0   
            else:
                Alive_symbol = 'Alive'

            censor_list.append(single_df['vital_status'] == Alive_symbol)
            surv_time = single_df['last_contact_days_to'] if single_df['vital_status'] == Alive_symbol else single_df['death_days_to']
            surv_time = int(surv_time)
            time_list.append(surv_time)
            # import ipdb; ipdb.set_trace()
            hazard_list.append(convert_time_to_period(surv_time, [t0, t1, t2, t3]))

            # for test
            # if len(feat_list) > 10:
            #     break

        self.name_list = data_names
        self.feat_list = feat_list
        self.omic_list = omics_list
        self.censor_list = censor_list
        self.time_list = time_list
        self.hazard_list = hazard_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        name = self.name_list[index]
        img_feat = torch.tensor(self.feat_list[index], dtype=torch.float32).squeeze()
        omic_feat = torch.tensor(self.omic_list[index], dtype=torch.float32).squeeze()
        censor = torch.tensor(self.censor_list[index], dtype=torch.float32)
        hazard = torch.tensor(self.hazard_list[index], dtype=torch.long)
        time = torch.tensor(self.time_list[index], dtype=torch.float32) / 30
        return {
            'name': name,
            'omic_feat': omic_feat,
            'img_feat': img_feat,
            'censor': censor,
            'time': hazard,
            'months': time
        }
    
class SurvivalImageDataset(Dataset):
    def __init__(self, feat_dir, omic_dir=None, survival_pth='', mode='TNBC'):    
        surv_df = pd.read_csv(survival_pth, index_col=0)
        t0 = 0
        death_days = np.array(surv_df['death_days_to'].dropna(), dtype=np.int32)
        t1, t2, t3 = np.quantile(death_days, q=0.25), np.median(death_days), np.quantile(death_days, q=0.75)
        censor_list, time_list, hazard_list = [], [], []

        data_names = []
        omics_list, feat_list = [], []
        for feat_pth in glob(os.path.join(feat_dir, '*.pkl')):

            if mode in ['TNBC']:
                slide_name = feat_pth.split('/')[-2]
                slide_file = glob(f'/data/gcf22/Spatial/BRCA/TNBC_ST/Images/imagesHD/*_{slide_name}.jpg')[0]
                slide_id = int(slide_file.split('/')[-1].split('_')[0].lstrip('TNBC'))

            else:
                slide_name = feat_pth.split('/')[-1].split('.pkl')[0]
                slide_id = slide_name
                
            if slide_id not in surv_df.index:
                print(f"{slide_id} not in clinical dataframe.")
                continue

            if pd.isna(surv_df.loc[slide_id]['vital_status']):
                continue

            if omic_dir is None:
                omics_list.append(0)
            else:
                pass
            data_names.append(slide_name)
            feat_list.append(feat_pth)
            single_df = surv_df.loc[slide_id]
            if mode in ['TNBC']:
                Alive_symbol = 0   
            else:
                Alive_symbol = 'Alive'

            censor_list.append(single_df['vital_status'] == Alive_symbol)
            surv_time = single_df['last_contact_days_to'] if single_df['vital_status'] == Alive_symbol else single_df['death_days_to']
            surv_time = int(surv_time)
            time_list.append(surv_time)
            # import ipdb; ipdb.set_trace()
            hazard_list.append(convert_time_to_period(surv_time, [t0, t1, t2, t3]))

        self.name_list = data_names
        self.feat_list = feat_list
        self.omic_list = omics_list
        self.censor_list = censor_list
        self.time_list = time_list
        self.hazard_list = hazard_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        name = self.name_list[index]
        feat_path = self.feat_list[index]
        with open(feat_path, 'rb') as file:
            feat_dict = pkl.load(file)

        img_feat = torch.tensor(np.stack(list(feat_dict.values())), dtype=torch.float32).squeeze()
        omic_feat = torch.tensor(self.omic_list[index], dtype=torch.float32).squeeze()
        censor = torch.tensor(self.censor_list[index], dtype=torch.float32)
        hazard = torch.tensor(self.hazard_list[index], dtype=torch.long)
        time = torch.tensor(self.time_list[index], dtype=torch.float32) / 30
        return {
            'name': name,
            'omic_feat': omic_feat,
            'img_feat': img_feat,
            'censor': censor,
            'time': hazard,
            'months': time
        }
    

class FusionImageDataset(Dataset):
    def __init__(self, img_dir, prefixs=None, survival_pth='') -> None:
        super().__init__()
        if prefixs is None:
            img_pths = glob(os.path.join(img_dir, '*.h5'))

        else:
            img_pths = [os.path.join(img_dir, f'{prefix}.h5') for prefix in prefixs]

        surv_df = pd.read_csv(survival_pth, index_col=0)
        surv_df = surv_df[surv_df['vital_status'] != '[Not Available]']
        sub_df = surv_df[surv_df['vital_status'] == 'Dead']
        t0 = 0
        death_days = np.array(sub_df['death_days_to'], dtype=np.int32)
        t1, t2, t3 = np.quantile(death_days, q=0.25), np.median(death_days), np.quantile(death_days, q=0.75)

        name_list = []
        censor_list, time_list, hazard_list = [], [], []

        for img_pth in img_pths:
            prefix = img_pth.split('/')[-1].split('.h5')[0]
            sample_id = prefix[:12]
            sample_name = prefix[:16]
            if sample_id not in surv_df.index:
                continue
            
            name_list.append(sample_name)
            single_df = surv_df.loc[[sample_id]].iloc[0]
            censor_list.append(single_df['vital_status'] == 'Alive')
            surv_time = single_df['last_contact_days_to'] + single_df['death_days_to'] + 1
            time_list.append(surv_time)
            hazard_list.append(convert_time_to_period(surv_time, [t0, t1, t2, t3]))

        self.name_list = name_list
        self.feat_list = img_pths
        self.censor_list = censor_list
        self.time_list = time_list
        self.hazard_list = hazard_list

    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, index):
        name = self.name_list[index]
        img_pth = self.feat_list[index]
        with h5py.File(img_pth, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
        img_feat = torch.tensor(features, dtype=torch.float32).squeeze()
        censor = torch.tensor(self.censor_list[index], dtype=torch.float32)
        hazard = torch.tensor(self.hazard_list[index], dtype=torch.long)
        time = torch.tensor(self.time_list[index], dtype=torch.float32) / 30
        return {
            'name': name,
            'img_feat': img_feat,
            'censor': censor,
            'time': hazard,
            'months': time
        }