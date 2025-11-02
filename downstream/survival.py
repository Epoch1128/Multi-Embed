import sys
import torch
import numpy as np
import os
import argparse
import pickle as pkl
from torch.optim import Adam
from torch import nn
from fusion import MultiSurvFix, NLLSurvLoss_dep
from data import FusionDataset, FusionRawDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
# from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored

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
    for idx, data in enumerate(train_bar):
        img_feat, omics_feat = data['img_feat'].to(device).squeeze(), data['omic_feat'].to(device)
        censor, surv_gt = data['censor'].to(device), data['time'].to(device)
        surv_pred = model(img_feat, omics_feat)
        hazards = torch.sigmoid(surv_pred)
        S = torch.cumprod(1 - hazards, dim=1)
        loss_out = loss_fn(hazards, S, surv_gt, censor, alpha=0.15)
        loss_out.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_value = loss_out.item()
        train_bar.set_description('epoch:{} iter:{} loss:{}'.format(epoch, idx, round(loss_value, 4)))
        loss_list.append(loss_value)
    return np.mean(loss_list)


def val_loop(epoch, model, dataloader, device):
    all_risk_scores, all_event_times, all_censorships = [], [], []
    all_names = []
    for idx, data in enumerate(dataloader):
        img_feat, omics_feat = data['img_feat'].to(device).squeeze(), data['omic_feat'].to(device)
        surv_pred = model(img_feat, omics_feat)
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
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return c_index, [all_names, all_risk_scores, all_censorships, all_event_times]
    

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(description="Prediction with Spots Images")
    parser.add_argument('--seed', default=47, type=int)
    # exp
    parser.add_argument('--exp', type=str, default='Coembedding')
    parser.add_argument('--save_model', action='store_true')

    # data
    # parser.add_argument('--data_type', type=str, default='Coembedding', choices=['ST', 'TCGA', 'Image-only', 'Coembedding'])
    parser.add_argument('--feat_dir', type=str, help='Directory to the image features')
    parser.add_argument('--omics_dir', type=str, help='Directory to the omics features')
    parser.add_argument('--survival_pth', type=str, help='Directory to the survival features')
    parser.add_argument('--save_dir', type=str, help='Directory to save the model')
    parser.add_argument('--prefix', type=str, help='Directory to prefix file')
    
    # train
    parser.add_argument('--model_desc', type=str, help='Model description', default='Baseline')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)

    # model
    parser.add_argument('--omics_dim', type=int, help='Dim of omic data feature', default=1024)
    parser.add_argument('--feat_dim', type=int, help='Dim of image data feature', default=256)
    # parser.add_argument('--hidden_dim', type=int, help='Dim of image data feature', default=512)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    prefix_info = np.load(args.prefix, allow_pickle=True)
    train_prefix, val_prefix = list(prefix_info[0]), list(prefix_info[1])

    if args.exp in ['Multi-Embed']:
        train_data = FusionDataset(args.feat_dir, prefixs=train_prefix, survival_pth=args.survival_pth)
        val_data = FusionDataset(args.feat_dir, prefixs=val_prefix, survival_pth=args.survival_pth)
    else:
        train_data = FusionRawDataset(args.feat_dir, args.omics_dir, prefixs=train_prefix, survival_pth=args.survival_pth)
        val_data = FusionRawDataset(args.feat_dir, args.omics_dir, prefixs=val_prefix, survival_pth=args.survival_pth)
        

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

    model = MultiSurvFix(input_dim=args.feat_dim, omics_dim=args.omics_dim)
    # print(model)
    opt = Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.is_available():
        print("Using 1 gpu")
    else:
        print("Using cpu")

    c_index_max = 0
    for iter in range(0, 50):
        mean_loss = train_loop(iter, model, train_loader, opt, device)
        # print(mean_loss)
        c_index, val_results = val_loop(iter, model, val_loader, device)
        # print(c_index)
        if c_index > c_index_max:
            c_index_max = c_index
            print(c_index)
            if args.save_model:
                save_checkpoint(model, opt, os.path.join(args.save_dir, f'best.ckpt'))
            with open(os.path.join(args.save_dir, f'best.pkl'), 'wb') as file:
                pkl.dump(
                    {
                        'c_index': c_index,
                        'results': val_results
                    }, file)   
    
if __name__ == '__main__':
    main()