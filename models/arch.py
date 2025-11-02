import torch
from models.model_utils import SNN_Block, Reg_Block
from torch import nn


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

class GeneEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size=[512, 256]):
        super(GeneEmbedding, self).__init__()
        self.fc_omic = nn.Sequential(
            SNN_Block(input_dim, hidden_size[0], dropout=0.25),
            SNN_Block(hidden_size[0], hidden_size[1], dropout=0.25)
        )

    def forward(self, x):
        gene_embedding = self.fc_omic(x)
        return gene_embedding
    
class ImageEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size=[512, 256]) -> None:
        super(ImageEmbedding, self).__init__()
        self.fc_image = nn.Sequential(
            Reg_Block(input_dim, hidden_size[0], dropout=0.25),
            Reg_Block(hidden_size[0], hidden_size[1], dropout=0.25)
        )
        self.attention_net = Attn_Net_Gated(L = hidden_size[1], D = hidden_size[1], dropout = 0.25, n_classes = 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_att=False):
        x = self.fc_image(x)
        A, x = self.attention_net(x)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = self.softmax(A)  # softmax over N
        chl, width, _ = A.shape
        if chl == 1:
            image_embedding = x
        elif width == 1:
            image_embedding = torch.matmul(A.squeeze(), x.squeeze(0)).unsqueeze(0).unsqueeze(1)            
        else:
            raise ValueError
        
        if return_att:
            return x, image_embedding, A
        else:
            return x, image_embedding, None
    
class GeneRecover(nn.Module):
    def __init__(self, output_dim, hidden_size=[512, 256]):
        super(GeneRecover, self).__init__()
        self.fc_omic = nn.Sequential(
            Reg_Block(hidden_size[0], hidden_size[1], dropout=0.25),
            nn.Linear(hidden_size[1], output_dim)
        )

    def forward(self, x):
        gene_expression = self.fc_omic(x)
        return gene_expression
    
class ImageRecover(nn.Module):
    def __init__(self, output_dim, hidden_size=[512, 256]) -> None:
        super(ImageRecover, self).__init__()
        self.fc_image = nn.Sequential(
            Reg_Block(hidden_size[0], hidden_size[1], dropout=0.25),
            nn.Linear(hidden_size[1], output_dim)
        )

    def forward(self, x):
        image_feature = self.fc_image(x)
        return image_feature
    
class MultiEmbed(nn.Module):
    def __init__(self, omic_dim, img_dim, hidden_size=[512, 256], shared_dim=256):
        super(MultiEmbed, self).__init__()
        self.gb = GeneEmbedding(omic_dim, hidden_size)
        self.ib = ImageEmbedding(img_dim, hidden_size)
        # self.shared_dim = shared_dim
        assert hidden_size[-1] >= shared_dim, 'The last dimension of hidden size should >= shared dimension'
        self.share = nn.Sequential(
            nn.Linear(hidden_size[-1], shared_dim),
            nn.ReLU()
        )
        self.gr = GeneRecover(omic_dim, hidden_size[::-1])
        self.ir = GeneRecover(img_dim, hidden_size[::-1])

    def gene_ae(self, omic_feat):
        omic_emb = self.gb(omic_feat)
        shared_omic_emb = self.share(omic_emb)
        omic_recon = self.gr(shared_omic_emb)
        return shared_omic_emb, omic_recon
    
    def image_ae(self, img_feat, return_att=False):
        img_emb, img_agg, att = self.ib(img_feat, return_att)
        shared_img_emb = self.share(img_emb)
        img_recon = self.ir(shared_img_emb)

        shared_img_agg = self.share(img_agg)
        return img_emb, img_recon, shared_img_agg, att

    def forward(self, img_feat=None, omic_feat=None, neg_feat=None, return_att=False):
        omic_emb, omic_recon, img_emb, img_recon, neg_img_emb, neg_img_recon, img_agg, neg_img_agg, img_att = \
            None, None, None, None, None, None, None, None, None
        if omic_feat is not None:
            omic_emb, omic_recon = self.gene_ae(omic_feat)

        if img_feat is not None:
            img_emb, img_recon, img_agg, img_att = self.image_ae(img_feat, return_att)
            
        if neg_feat is not None:
            neg_img_emb, neg_img_recon, neg_img_agg, _ = self.image_ae(neg_feat)

        return omic_emb, img_emb, neg_img_emb, omic_recon, img_recon, neg_img_recon, img_agg, neg_img_agg, img_att