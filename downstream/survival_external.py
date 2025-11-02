
import pandas as pd
import torch
import numpy as np
import os
import argparse
import pickle as pkl
from torch.optim import Adam
from torch import nn
from fusion import MultiSurvFix, NLLSurvLoss_dep, AMIL
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.utils.data import Subset
# from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored


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
    def __init__(self, feat_pth, survival_pth=''):
        with open(feat_pth, 'rb') as file:
            feat_dict = pkl.load(file)
        feat_list = []
        agg_list = []
            
        surv_df = pd.read_csv(survival_pth, index_col=0).dropna()
        surv_df = surv_df[surv_df['vital_status'] != '[Not Available]']
        sub_df = surv_df[surv_df['vital_status'] == 'Dead']
        t0 = 0
        death_days = np.array(sub_df['death_days_to'], dtype=np.int32)
        t1, t2, t3 = np.quantile(death_days, q=0.25), np.median(death_days), np.quantile(death_days, q=0.75)
        censor_list, time_list, hazard_list = [], [], []
        name_list = []
        
        for name_idx, sample_name in enumerate(feat_dict['names']):
            sample_id = sample_name.split('.svs')[0]

            slide_id = sample_id[:12] if sample_id.startswith('TCGA') else sample_id
            if slide_id not in surv_df.index:
                continue
            name_list.append(slide_id)
            agg_list.append(feat_dict['agg_images'][name_idx])
            feat_list.append(feat_dict['images'][name_idx])
            single_df = surv_df.loc[[slide_id]].iloc[0]
            censor_list.append(single_df['vital_status'] == 'Alive')
            surv_time = single_df['last_contact_days_to'] + single_df['death_days_to'] + 1
            time_list.append(surv_time)
            hazard_list.append(convert_time_to_period(surv_time, [t0, t1, t2, t3]))

        self.name_list = name_list
        self.feat_list = feat_list
        self.agg_list = agg_list
        self.censor_list = censor_list
        self.time_list = time_list
        self.hazard_list = hazard_list

    def __len__(self):
        return len(self.feat_list)
    
    def __getitem__(self, index):
        item_dict = {}
        name = self.name_list[index]
        img_feat = torch.tensor(self.feat_list[index], dtype=torch.float32).squeeze()
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
    

def save_checkpoint(model, optimizer, save_dir):
    print(f"Saving checkpoint to {save_dir}")
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_dir)

def train_loop(epoch, model, dataloader, optimizer, device):
    loss_fn = NLLSurvLoss_dep()
    train_bar = tqdm(dataloader, desc="epoch " + str(epoch), total=len(dataloader),
                            unit="batch", dynamic_ncols=True)
    loss_list = []
    all_risk_scores = []
    for idx, data in enumerate(train_bar):
        images_feat = data['img_feat'].to(device)
        censor, surv_gt = data['censor'].to(device), data['time'].to(device)
        surv_pred = model(images_feat).squeeze(1)
        hazards = torch.sigmoid(surv_pred)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores.append(risk)

        loss_out = loss_fn(hazards, S, surv_gt, censor, alpha=0.15)
        loss_out.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_value = loss_out.item()
        train_bar.set_description('epoch:{} iter:{} loss:{}'.format(epoch, idx, round(loss_value, 4)))
        loss_list.append(loss_value)
    print(f"Cutoff: {np.median(all_risk_scores)}.")
    return np.mean(loss_list)

def val_loop(epoch, model, dataloader, device):
    all_risk_scores, all_event_times, all_censorships = [], [], []
    all_names = []
    for data in tqdm(dataloader):
        images_feat = data['img_feat'].to(device)
        surv_pred = model(images_feat).squeeze(1)
        hazards = torch.sigmoid(surv_pred)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        
        all_risk_scores.append(risk)
        all_event_times.append(data['months'].numpy())
        all_censorships.append(data['censor'].numpy())
        all_names.append(data['name'])

    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    all_names = np.concatenate(all_names)

    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
    # import ipdb; ipdb.set_trace()
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return c_index, [all_names, all_risk_scores, all_censorships, all_event_times]
    

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(description="Prediction with Spots Images")
    parser.add_argument('--seed', default=47, type=int)
    parser.add_argument('--train_feat_dir', type=str, help='Directory to the train image features', default=None)
    parser.add_argument('--train_survival_pth', type=str, help='Directory to the train omics features', default=None)

    parser.add_argument('--feat_dir', type=str, help='Directory to the image features')
    parser.add_argument('--omics_dir', type=str, help='Directory to the omics features')
    parser.add_argument('--survival_pth', type=str, help='Directory to the survival features')
    parser.add_argument('--save_dir', type=str, help='Directory to save the model')

    # model
    parser.add_argument('--omics_dim', type=int, help='Dim of omic data feature', default=1024)
    parser.add_argument('--feat_dim', type=int, help='Dim of image data feature', default=256)

    # test
    parser.add_argument('--checkpoint', type=str, help='Pre-trained model for evaluation', default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print('>>> Loading data...')
    val_data = FusionDataset(
        args.feat_dir,
        survival_pth=args.survival_pth
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print("Using 1 gpu")
    else:
        print("Using cpu")

    model = AMIL(size_arg='big')
    
    train_mode = True
    if args.checkpoint is not None:
        params = torch.load(args.checkpoint)
        model.load_state_dict(params['state_dict'])
        print('>>> Start test...')
        train_mode = False

    else:
        train_data = FusionDataset(
            args.train_feat_dir, 
            survival_pth=args.train_survival_pth
            )
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        print(">>> Start training...")
        opt = Adam(model.parameters(), lr=1e-5)

    model.to(device)

    if train_mode:
        for iter in range(0, 50):
            mean_loss = train_loop(iter, model, train_loader, opt, device)
        
        save_checkpoint(model, opt, os.path.join(args.save_dir, f'prognosis.ckpt'))
    
    c_index, val_results = val_loop(0, model, val_loader, device)
    
    print(f">>> c-index for prognosis prediction: {c_index}")
            
    with open(os.path.join(args.save_dir, f'res.pkl'), 'wb') as file:
        pkl.dump(
            {
                'c_index': c_index,
                'results': val_results
            }, file)

if __name__ == '__main__':
    main()