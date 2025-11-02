import sys
sys.path.append('./')
import torch
import os
import argparse
from torch.nn.functional import normalize
from data.dataset import STDataset, BulkDataset, VisiumDataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.arch import MultiEmbed
from models.loss import SetTripletLoss, Tripletloss, mmd_rbf_loss, recon_loss, smooth_chamfer_distance, smooth_chamfer_distance
from utils import setup_seed, save_checkpoint, get_prefix
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Prediction with Spots Images")
    parser.add_argument('--seed', default=47, type=int)
    parser.add_argument('--gpu', default='1', type=str)

    # data
    parser.add_argument('--data_type', type=str, default='TCGA')
    parser.add_argument('--image_dir', type=str, help='Directory to the image features')
    parser.add_argument('--omics_dir', type=str, help='Directory to the omics features')
    parser.add_argument('--save_dir', type=str, help='Directory to save the model')
    parser.add_argument('--prefix', required = False, nargs = '+', default=None)
    
    # train
    parser.add_argument('--model_desc', type=str, help='Model description', default='Baseline')
    parser.add_argument('--batch', type=int, help='Batch size', default=32)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--epoch', type=int, help='Training epoch', default=250)

    # model
    parser.add_argument('--trained_model', type=str, default=None, help="model parameters for training")
    parser.add_argument('--omics_dim', type=int, help='Dim of omic data feature', default=1024)
    parser.add_argument('--image_dim', type=int, help='Dim of image data feature', default=1024)
    parser.add_argument('--shared_dim', type=int, help='Dim of multimodal data shared feature', default=256)
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    setup_seed(args.seed)
    print(f"Model details: {args.model_desc}")
    
    print("Load Dataset...")

    if args.prefix is None:
        data_prefix = None
        print('Load all data')
    else:
        data_prefix = get_prefix(args.prefix)
        # print(f'Load {data_prefix}')

    if args.data_type == 'TCGA':
        train_data = BulkDataset(args.image_dir, args.omics_dir, prefixs=data_prefix)
        print(f"length of train data: {len(train_data)}")
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    else:
        raise NotImplementedError("Use train_st.py to train with ST data.")
    
    print("Load Model...")
    model = MultiEmbed(omic_dim=args.omics_dim, img_dim=args.image_dim, hidden_size=[512, 512], shared_dim=args.shared_dim)

    if args.trained_model is not None:
        model_state_dict = torch.load(args.trained_model)['state_dict']
        model.load_state_dict(model_state_dict)

    optimizer = Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using 1 gpu")
    else:
        print("Using cpu")

    model.to(device)

    print("Training...")
    loss_min = 1e10
    save_dir = os.path.join(args.save_dir, args.model_desc)
    os.makedirs(save_dir, exist_ok=True)
    for iter in range(0, args.epoch):
        loss_value = train_loop(args, iter, model, train_loader, optimizer, device)
        if loss_value < loss_min and loss_value < 2:
            loss_min = loss_value
            save_checkpoint(model, optimizer, os.path.join(save_dir, f'best_{iter}_{round(loss_value, 2)}.ckpt'))
    
    print(f'Min loss: {loss_min}')
    print("Training finished!")


def train_loop(args, epoch, model, train_loader, opt, device):
    model.train()
    train_bar = tqdm(train_loader, desc="epoch " + str(epoch), total=len(train_loader),
                            unit="batch", dynamic_ncols=True)
    loss_sum, bnum = 0, 0
    for idx, data in enumerate(train_bar):
        img_feat, omic_feat, neg_feat = data['image_feat'].to(device), data['omic_feat'].to(device), data['neg_feat'].to(device)
        omic_emb, img_emb, neg_img_emb, omic_recon, img_recon, neg_img_recon, img_agg, neg_img_agg, _ = \
            model(img_feat, omic_feat, neg_feat)

        triple_loss = SetTripletLoss(normalize(img_emb, p=2, dim=-1), normalize(neg_img_emb, p=2, dim=-1), normalize(omic_emb, p=2, dim=-1), margin=1, distance_fn=smooth_chamfer_distance, dist_choice='cosine')
        agg_triple_loss = Tripletloss(normalize(omic_emb, p=2, dim=-1), normalize(img_agg, p=2, dim=-1), normalize(neg_img_agg, p=2, dim=-1))
        rloss = recon_loss(omic_recon, omic_feat) * 0.05 + recon_loss(img_recon, img_feat) + recon_loss(neg_img_recon, neg_feat)
        rbf_loss = mmd_rbf_loss(omic_emb, img_emb)
        loss_out = rloss + rbf_loss + (triple_loss + agg_triple_loss) * args.alpha
        loss_out.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad()

        loss_value = loss_out.item()
        loss_sum += loss_value
        bnum += 1
        train_bar.set_description('epoch:{} iter:{} all:{} tri:{} tri-agg:{}'.format(epoch, idx, round(loss_value, 3), round(triple_loss.item(), 3), round(agg_triple_loss.item(), 3)))
    return loss_sum / bnum

if __name__ == '__main__':
    main()
