import torch
from models.model_utils import SNN_Block, Reg_Block
from torch import nn
from typing import Dict, Optional, Any

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_size=[512, 256]):
        super(Encoder, self).__init__()
        self.fc_omic = nn.Sequential(
            SNN_Block(input_dim, hidden_size[0], dropout=0.25),
            SNN_Block(hidden_size[0], hidden_size[1], dropout=0.25)
        )

    def forward(self, x):
        gene_embedding = self.fc_omic(x)
        return gene_embedding

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_size=[512, 256]):
        super(Decoder, self).__init__()
        self.fc_omic = nn.Sequential(
            Reg_Block(hidden_size[0], hidden_size[1], dropout=0.25),
            nn.Linear(hidden_size[1], output_dim)
        )

    def forward(self, x):
        gene_expression = self.fc_omic(x)
        return gene_expression
    
class MultiEmbedST(nn.Module):
    def __init__(
        self,
        feature_dims: Dict[str, int],
        hidden_size = [512, 256],
        shared_dim: int = 256,
    ):
        super(MultiEmbedST, self).__init__()
        assert hidden_size[-1] >= shared_dim, \
            "The last dimension of hidden_size should be >= shared_dim"

        self.feature_dims = feature_dims

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for name, dim in feature_dims.items():
            self.encoders[name] = Encoder(dim, hidden_size)
            self.decoders[name] = Decoder(dim, hidden_size[::-1])
            
        self.share = nn.Sequential(
            nn.Linear(hidden_size[-1], shared_dim),
            nn.ReLU()
        )

    @staticmethod
    def _assert_batch_sizes(anchor_feat: torch.Tensor,
                            other_features: Dict[str, torch.Tensor],
                            neg_anchor_feat: Optional[torch.Tensor] = None):
        sizes = [anchor_feat.shape[0]]
        for k, v in other_features.items():
            sizes.append(v.shape[0])
        if neg_anchor_feat is not None:
            sizes.append(neg_anchor_feat.shape[0])

        unique_sizes = set(sizes)
        assert len(unique_sizes) == 1, \
            f"All inputs must share the same batch size, got: {unique_sizes}"

    def encode_single(self, modality: str, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoder = self.encoders[modality]
        decoder = self.decoders[modality]

        emb = encoder(x)             # (B, hidden_size[-1])
        shared_emb = self.share(emb) # (B, shared_dim)
        recon = decoder(shared_emb)  # (B, input_dim)

        return {
            "embed": shared_emb,
            "recon": recon,
        }
    
    def forward(
        self,
        anchor_mod: str,
        anchor_feat: torch.Tensor,
        other_features: Dict[str, torch.Tensor],
        neg_anchor_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        self._assert_batch_sizes(anchor_feat, other_features, neg_anchor_feat)
        anchor_out = self.encode_single(anchor_mod, anchor_feat)
        neg_out = None
        if neg_anchor_feat is not None:
            neg_out = self.encode_single(anchor_mod, neg_anchor_feat)
        others_out: Dict[str, Dict[str, torch.Tensor]] = {}
        for mod_name, feat in other_features.items():
            out = self.encode_single(mod_name, feat)
            others_out[mod_name] = out

        return {
            "anchor": {
                "modality": anchor_mod,
                **anchor_out,
            },
            "neg_anchor": neg_out,
            "others": others_out,
        }
