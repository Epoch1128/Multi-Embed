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


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

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
    corr, corr_df = calc_corr(omics_mat, omics_gt)
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
    return corr, corr_df, save_info


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser(description="Prediction with Spots Images")
    parser.add_argument('--seed', default=47, type=int)
    parser.add_argument('--gpu', default='0', type=str)

    # data
    parser.add_argument('--image_dir', type=str, help='Directory to the image features')
    parser.add_argument('--omics_dir', type=str, help='Directory to the omics features')
    parser.add_argument('--save_dir', type=str, help='Directory to save the results')
    parser.add_argument('--prefix', type=str, help='Directory to prefix file')
    parser.add_argument('--save_tiles', action='store_true', help='Whether to save tiles')
    parser.add_argument('--save_models', action='store_true', help='Whether to save models')
    
    # train
    parser.add_argument('--model_desc', type=str, help='Model description', default='Baseline')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--epoch', type=int, help='Training epoch', default=30)

    # model
    parser.add_argument('--omics_dim', type=int, help='Dim of omic data feature', default=1024)
    parser.add_argument('--image_dim', type=int, help='Dim of image data feature', default=256)
    parser.add_argument('--hidden_dim', type=int, help='Dim of image data feature', default=512)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    save_dir = os.path.dirname(os.path.abspath(args.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    prefix_info = np.load(args.prefix, allow_pickle=True)
    train_prefix, val_prefix = list(prefix_info[0]), list(prefix_info[1])

    print('Loading data...')
    train_data = RNADataset(args.image_dir, args.omics_dir, prefixs=train_prefix)
    val_data = RNADataset(args.image_dir, args.omics_dir, prefixs=val_prefix)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    model = Emb2RNA(input_dim=args.image_dim, output_dim=args.omics_dim, hidden_size=args.hidden_dim)
    opt = Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.is_available():
        print("Using 1 gpu")
    else:
        print("Using cpu")

    corr_max = 0
    for iter in range(0, args.epoch):
        mean_loss = train_loop(iter, model, train_loader, opt, device)
        if iter % 10 == 0:
            print(f">>> Iter at {iter}")
            print(f">>> Average loss: {mean_loss}")
            print(f">>> Average correlation: {corr_max}")

        mean_corr, corr_df, save_info = val_loop(iter, model, val_loader, device, save_tiles=args.save_tiles)
        if mean_corr > corr_max:
            corr_max = mean_corr
            if args.save_models:
                torch.save(model.state_dict(), args.save_model_dir)
                print(f">>> Model has been saved at {args.save_model_dir}")
            with open(args.save_dir, 'wb') as file:
                pkl.dump(save_info, file)
    print(f">>> Prediction results have been saved at {args.save_dir}")
    print(f">>> Final average correlation: {corr_max}")
    
if __name__ == '__main__':
    main()