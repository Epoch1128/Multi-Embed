import torch
import numpy as np
import pandas as pd
import pickle as pkl
from torch.utils.data import Dataset

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
        name_list = []
            
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
            name_list.append(sample_name)
            agg_list.append(feat_dict['agg_images'][name_idx])
            feat_list.append(feat_dict['images'][name_idx])
            single_df = surv_df.loc[[sample_id]].iloc[0]
            censor_list.append(single_df['vital_status'] == 'Alive')
            surv_time = single_df['last_contact_days_to'] + single_df['death_days_to'] + 1
            time_list.append(surv_time)
            hazard_list.append(convert_time_to_period(surv_time, [t0, t1, t2, t3]))

        self.name_list = name_list
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