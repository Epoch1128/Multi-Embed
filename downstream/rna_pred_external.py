import torch
import numpy as np
import os
import argparse
import pickle as pkl
from torch.optim import Adam
from torch import nn
from he2rna import calc_corr
from data import RNADataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class Emb2RNA(nn.Module):
    def __init__(self, input_dim, output_dim=1024, hidden_size=512) -> None:
        super(Emb2RNA, self).__init__()
        self.layer0 = nn.Sequential(
            #nn.Conv1d(in_channels=n_inputs, out_channels=n_hiddens,kernel_size=1, stride=1, bias=True),
            nn.Linear(input_dim, hidden_size),
            #nn.ReLU(),  ## 2020.03.26: for positive gene expression
            nn.Dropout(0.5)
            )
        self.layer1 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        y = self.layer0(x)
        y = self.layer1(y)
        gene = y.mean(dim=1)
        return gene, y

def train_loop(epoch, model, dataloader, optimizer, device):
    loss_fn = nn.MSELoss()
    train_bar = tqdm(dataloader, desc="epoch " + str(epoch), total=len(dataloader),
                            unit="batch", dynamic_ncols=True)
    loss_list = []
    for idx, data in enumerate(train_bar):
        img_feat, omics_lbl = data['img_feat'].to(device), data['omic_feat'].to(device)
        omics_pred, _ = model(img_feat)
        loss_out = loss_fn(omics_lbl, omics_pred)
        loss_out.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_value = loss_out.item()
        train_bar.set_description('epoch:{} iter:{} loss:{}'.format(epoch, idx, round(loss_value, 4)))
        loss_list.append(loss_value)
    return np.mean(loss_list)


def val_loop(epoch, model, dataloader, device, save_tiles=False):
    pred_list, gt_list = [], []
    pred_tiles_list = []
    name_list = []
    for data in dataloader:
        img_feat = data['img_feat'].to(device)
        omics_pred, omics_st = model(img_feat)
        omics_pred = omics_pred.squeeze().detach().cpu().numpy()
        pred_list.append(omics_pred)
        gt_list.append(data['omic_feat'].squeeze())
        name_list.extend(data['name'])
        if save_tiles:
            omics_st = omics_st.squeeze().detach().cpu().numpy()
            pred_tiles_list.append(omics_st)

    omics_mat = np.stack(pred_list, axis=0)
    omics_gt = np.stack(gt_list, axis=0)
    mean_corr, corr_df = calc_corr(omics_mat, omics_gt)
    if save_tiles:
        save_info = {
            'names': name_list,
            'pred': omics_mat,
            'gt': omics_gt,
            'pred_tiles': pred_tiles_list
        }

    else:
        save_info = {
            'names': name_list,
            'pred': omics_mat,
            'gt': omics_gt
        }
    return mean_corr, corr_df, save_info


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser(description="Prediction with Spots Images")
    parser.add_argument('--seed', default=47, type=int)
    parser.add_argument('--gpu', default='0', type=str)

    # data
    parser.add_argument('--image_dir', type=str, help='Directory to the image features', default=None)
    parser.add_argument('--omics_dir', type=str, help='Directory to the omics features', default=None)

    parser.add_argument('--val_image_dir', type=str, help='Directory to the image features')
    parser.add_argument('--val_omics_dir', type=str, help='Directory to the omics features')

    parser.add_argument('--save_dir', type=str, help='Directory to save the results', default=None)
    parser.add_argument('--save_model_dir', type=str, help='Directory to save the model', default=None)
    parser.add_argument('--save_models', action='store_true', help='Whether to save models')
    parser.add_argument('--save_tiles', action='store_true', help='Whether to save gene expression results for each tile')
    
    # train
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)

    # test
    parser.add_argument('--checkpoint', type=str, help='Pre-trained model', default=None)

    # model
    parser.add_argument('--omics_dim', type=int, help='Dim of omic data feature', default=1024)
    parser.add_argument('--image_dim', type=int, help='Dim of image data feature', default=256)
    parser.add_argument('--hidden_dim', type=int, help='Dim of image data feature', default=512)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    save_dir = os.path.dirname(os.path.abspath(args.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    save_model_dir = os.path.dirname(os.path.abspath(args.save_model_dir))
    os.makedirs(save_model_dir, exist_ok=True)

    print('>>> Loading data...')
    
    val_data = RNADataset(args.val_image_dir, args.val_omics_dir, datatype='CPTAC')
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    model = Emb2RNA(input_dim=args.image_dim, output_dim=args.omics_dim, hidden_size=args.hidden_dim)

    train_mode = True
    if args.checkpoint is not None:
        params = torch.load(args.checkpoint)
        model.load_state_dict(params)
        print('>>> Start test...')
        train_mode = False

    else:
        train_data = RNADataset(args.image_dir, args.omics_dir)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        print(">>> Start training...")
        opt = Adam(model.parameters(), lr=args.lr)
        
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    if torch.cuda.is_available():
        print("Using 1 gpu")
    else:
        print("Using cpu")

    if train_mode:
        for iter in range(0, 200):
            mean_loss = train_loop(iter, model, train_loader, opt, device)
            
            if iter % 20 == 0:
                print(f">>> Average loss: {mean_loss}")

    mean_corr, corr_df, save_info = val_loop(0, model, val_loader, device, save_tiles=args.save_tiles)
    corr_df.index = val_data.gene_list
    print(f">>> Average correlation for genome-wide: {mean_corr}")

    import json
    with open("../save/TCGA-COAD/hvgs.json", "r") as f:
        gene_list = json.load(f)
    print(f">>> Average correlation for HVGs: {np.mean(corr_df.loc[gene_list].values)}")

    if args.save_models and train_mode:
        print(f">>> Model has been saved at {args.save_model_dir}")
        torch.save(model.state_dict(), args.save_model_dir)
    with open(args.save_dir, 'wb') as file:
        print(f">>> Prediction results have been saved at {args.save_dir}")
        save_info['corrs'] = corr_df
        pkl.dump(save_info, file)
    
if __name__ == '__main__':
    main()