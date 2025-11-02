from traitlets import default
import sys
sys.path.append('./')
import torch
import os
import argparse
import numpy as np
import pickle as pkl
from data.dataset import STDataset, BulkDataset, VisiumDataset, XeniumDataset
from torch.utils.data import DataLoader
from models.arch import MultiEmbed
from utils import setup_seed, get_prefix
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Prediction with Spots Images")
    parser.add_argument('--seed', default=47, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--agg_only', action='store_true')

    # data
    parser.add_argument('--data_type', type=str, default='ST')
    parser.add_argument('--image_dir', type=str, help='Directory to the image features')
    parser.add_argument('--omics_dir', type=str, help='Directory to the omics features')
    parser.add_argument('--save_dir', type=str, help='Directory to save the model')
    parser.add_argument('--save_name', type=str, help='Name to save the model', default='test')
    parser.add_argument('--cluster_dir', type=str, help='Directory to cluster')
    parser.add_argument('--prefix', required = False, nargs = '+')
    
    # train
    parser.add_argument('--model_pth', type=str, help='Model description')
    parser.add_argument('--batch', type=int, help='Batch size', default=32)

    # model
    parser.add_argument('--omics_dim', type=int, help='Dim of omic data feature', default=1024)
    parser.add_argument('--image_dim', type=int, help='Dim of image data feature', default=1024)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    setup_seed(args.seed)
    print("Load Dataset...")

    if args.prefix is None:
        data_prefix = None
        print('Load all data')
    else:
        data_prefix = get_prefix(args.prefix)
        print(f'Load {data_prefix}')

    if args.data_type == 'ST':
        train_data = STDataset(args.image_dir, args.omics_dir, prefixs=data_prefix)
        print(f"length of train data: {len(train_data)}")
        train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)

    elif args.data_type == 'Xenium':
        train_data = XeniumDataset(args.image_dir, args.omics_dir, cluster_dir=args.cluster_dir, prefixs=data_prefix)
        print(f"length of train data: {len(train_data)}")
        train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)

    elif args.data_type == 'VisiumHD':
        train_data = VisiumDataset(args.image_dir, args.omics_dir)
        print(f"length of train data: {len(train_data)}")
        train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=16, pin_memory=True)

    elif args.data_type == 'TCGA':
        train_data = BulkDataset(args.image_dir, args.omics_dir, prefixs=data_prefix, max_img=1e10)
        print(f"length of train data: {len(train_data)}")
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    
    elif args.data_type == 'image-only':
        pass

    else:
        raise NotImplementedError
    
    print("Load Model...")
    model = MultiEmbed(omic_dim=args.omics_dim, img_dim=args.image_dim, hidden_size=[512, 512], shared_dim=512)
    model_state_dict = torch.load(args.model_pth)['state_dict']
    model.load_state_dict(model_state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using 1 gpu")
    else:
        print("Using cpu")

    model.to(device)

    print("Infering...")
    os.makedirs(args.save_dir, exist_ok=True)
    val_loop(model, train_loader, device, save_path=os.path.join(args.save_dir, f"{args.save_name}_results.pkl"), agg_only=args.agg_only)

    print("Inference finished!")


def val_loop(model, train_loader, device, save_path, agg_only):
    model.eval()

    omic_list, img_list, name_list, agg_img_list = [], [], [], []
    attention_list = []
    for data in tqdm(train_loader):
        if 'image_feat' in data.keys():
            img_feat, omic_feat = data['image_feat'].to(device), data['omic_feat'].to(device)

        else:
            img_feat, omic_feat = data['protein_feat'].to(device), data['omic_feat'].to(device)
        
        omic_emb, img_emb, neg_img_emb, omic_recon, img_recon, neg_img_recon, img_agg, neg_img_agg, img_att = \
            model(img_feat, omic_feat, return_att=True)
        name_list.extend(data['name'])
        if not agg_only:
            img_list.append(img_emb.squeeze().detach().cpu().numpy())
        omic_list.append(omic_emb.detach().cpu().numpy())
        agg_img_list.append(img_agg.detach().cpu().numpy())
        attention_list.append(img_att.detach().cpu().numpy())

    img_mat = np.concatenate(agg_img_list, axis=0).squeeze()
    omic_mat = np.concatenate(omic_list, axis=0).squeeze()
    print('Saving results...')
    print(f'Length of image list: {len(img_list)}')
    with open(save_path, 'wb') as file:
        pkl.dump({'names': name_list, 'images': img_list, 'omics': omic_mat, 'agg_images': img_mat, 'attention': attention_list}, file)
        # pkl.dump({'names': name_list, 'omics': omic_mat, 'agg_images': img_mat}, file)
        
if __name__ == '__main__':
    main()
