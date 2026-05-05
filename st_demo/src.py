import sys
sys.path.append('./')
import torch
import random
import numpy as np
from torch import nn
from torch.nn.functional import normalize
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from models.arch import MultiEmbedST
from models.loss import Tripletloss, mmd_rbf_loss, recon_loss
from tqdm import tqdm
from typing import Dict, List, Optional, Union

class MultiOmicsDataset(Dataset):
    def __init__(
        self,
        features: Dict[str, Union[torch.Tensor]],
        names: Optional[List[str]] = None,
        anchor_key: str = "image",
    ):
        super().__init__()
        first_dim_set = {v.shape[0] for v in features.values()}
        assert len(first_dim_set) == 1, "Inconsistent first dimension among features"
        num_samples = list(first_dim_set)[0]
        if names is None:
            names = [f"Sample_{i}" for i in range(num_samples)]
        assert len(names) == num_samples, "Length of names must match number of samples"
        assert anchor_key in features, f"anchor_key '{anchor_key}' not found in features keys: {list(features.keys())}"
        self.anchor_key = anchor_key
        self.name_list = names

        self.anchor_features = features[anchor_key]
        self.other_keys = [k for k in features.keys() if k != anchor_key]
        self.other_features = {k: features[k] for k in self.other_keys}
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    @staticmethod
    def _generate_negative_index(except_index: int, min_val: int, max_val: int) -> int:
        while True:
            random_num = random.randint(min_val, max_val)
            if random_num != except_index:
                return random_num

    def __getitem__(self, index: int):
        neg_idx = self._generate_negative_index(
            except_index=index,
            min_val=0,
            max_val=self.num_samples - 1,
        )
        anchor_feat = torch.as_tensor(
            self.anchor_features[index],
            dtype=torch.float32,
        ).unsqueeze(0)
        neg_feat = torch.as_tensor(
            self.anchor_features[neg_idx],
            dtype=torch.float32,
        ).unsqueeze(0)
        other_feats = {
            key: torch.as_tensor(feats[index], dtype=torch.float32).unsqueeze(0)
            for key, feats in self.other_features.items()
        }

        return {
            "name": self.name_list[index],
            "anchor_feat": anchor_feat,
            "other_features": other_feats,
            "neg_feat": neg_feat,
        }

class Multi_Embed(nn.Module):
    def __init__(self, model_configs: dict, features: dict, anchor_keys='image', sample_names=None, device='cpu') -> None:
        super(Multi_Embed, self).__init__()
        self.device = device
        self.num_views = len(features)
        self.feature_names = list(features.keys())
        self.anchor = anchor_keys
        self.dataset = MultiOmicsDataset(features, sample_names, anchor_keys)
        self.dataloader = DataLoader(self.dataset, batch_size=32)

        hidden_size, embed_size, checkpoints = \
            model_configs['hidden_size'], model_configs['embed_size'], model_configs['checkpoints']
        self.model = MultiEmbedST(
            feature_dims={key: value.shape[1] for key, value in features.items()}, 
            hidden_size=hidden_size, 
            shared_dim=embed_size
        )
        
        if checkpoints is not None:
            model_state_dict = torch.load(checkpoints)
            self.model.load_state_dict(model_state_dict)

    def train(self, epochs=500, learning_rate=1e-4, beta=1):
        self.model.to(self.device)
        print("Start training...")
        loss_min = 1e10
        train_bar = tqdm(range(0, epochs), desc="epoch", total=epochs, unit="batch", dynamic_ncols=True)
        self.opt = Adam(self.model.parameters(), lr=learning_rate)
        self.beta = beta
        for iter in train_bar:
            loss_value = self.train_loop()
            if loss_min > loss_value:
                loss_min = loss_value
            train_bar.set_description('epoch: {}; loss: {}'.format(iter, round(loss_value, 3)))

        print(f'Min loss: {loss_min}')
        print("Training finished!")

    def train_loop(self):
        self.model.train()
        loss_sum, bnum = 0, 0
        for data in self.dataloader:
            anchor_feat, neg_anchor_feat = data['anchor_feat'].to(self.device), data['neg_feat'].to(self.device)
            other_feats = {key: value.to(self.device) for key, value in data['other_features'].items()}
            res_dict = \
                self.model(self.anchor, anchor_feat, other_feats, neg_anchor_feat)

            triple_loss = 0
            rloss = recon_loss(anchor_feat, res_dict['anchor']['recon']) + recon_loss(neg_anchor_feat, res_dict['neg_anchor']['recon'])
            rbf_loss = 0
            for feat_key, other_dict in res_dict['others'].items():
                triple_loss += Tripletloss(
                    normalize(res_dict['anchor']['embed'], p=2, dim=-1), 
                    normalize(other_dict['embed'], p=2, dim=-1), 
                    normalize(res_dict['neg_anchor']['embed'], p=2, dim=-1)
                )
                rbf_loss += mmd_rbf_loss(res_dict['anchor']['embed'], other_dict['embed'])

                meta_feature = other_feats[feat_key]
                rloss += recon_loss(meta_feature, other_dict['recon'])
            loss_out = rloss * 0.1 + rbf_loss + triple_loss * self.beta
            loss_out.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            self.opt.zero_grad()

            loss_value = loss_out.item()
            loss_sum += loss_value
            bnum += 1

        return loss_sum / bnum

    def get_embeddings(self, infer_features: dict):
        self.model.eval()
        all_features = {}
        with torch.no_grad():
            for mod_name, feat_mat in infer_features.items():
                assert mod_name in self.feature_names, \
                    f"{mod_name} is not included in the development of Multi-Embed"
                feat_list = []
                for feat in torch.tensor(feat_mat, dtype=torch.float32).to(self.device):
                    embed = self.model.encode_single(mod_name, feat.unsqueeze(0))["embed"]
                    feat_list.append(embed.detach().cpu().numpy())

                feat_mat = np.concatenate(feat_list, axis=0)
                all_features.update(
                    {
                        mod_name: feat_mat
                    }
                )
        return all_features
