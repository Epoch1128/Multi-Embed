import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

class LRBilinearFusion(nn.Module):
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, 
                 scale_dim1=1, scale_dim2=1, dropout_rate=0.25,
                rank=16, output_dim=4):
        super(LRBilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.rank = rank
        self.output_dim = output_dim

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        self.h1_factor = Parameter(torch.Tensor(self.rank, dim1 + 1, output_dim))
        self.h2_factor = Parameter(torch.Tensor(self.rank, dim2 + 1, output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))
        xavier_normal_(self.h1_factor)
        xavier_normal_(self.h2_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

        #init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = F.dropout(self.linear_h1(vec1), 0.25)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = F.dropout(self.linear_h2(vec2), 0.25)
            o2 = self.linear_o2(h2)

        ### Fusion
        DTYPE = torch.cuda.FloatTensor
        _o1 = torch.cat((torch.ones(1, 1).type(DTYPE).clone().detach().requires_grad_(True), o1), dim=1)
        _o2 = torch.cat((torch.ones(1, 1).type(DTYPE).clone().detach().requires_grad_(True), o2), dim=1)
        o1_fusion = torch.matmul(_o1, self.h1_factor)
        o2_fusion = torch.matmul(_o2, self.h2_factor)
        fusion_zy = o1_fusion * o2_fusion
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output


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

class MultiSurv(nn.Module):
    def __init__(self, input_dim, output_dim=4, hidden_dim=256):
        super(MultiSurv, self).__init__()
        self.img_att = Attn_Net_Gated(L=input_dim, D=hidden_dim)
        self.omic_emb = nn.Linear(input_dim, hidden_dim)
        self.fusion = LRBilinearFusion(dim1=hidden_dim, dim2=hidden_dim, output_dim=output_dim)

    def forward(self, imgs, omics):
        # imgs N x F omics F
        A_path, h_path = self.img_att(imgs)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)

        h_omic = self.omic_emb(omics)
        return self.fusion(h_path, h_omic)
    

class MultiSurvFix(nn.Module):
    def __init__(self, input_dim, output_dim=4, hidden_dim=256, omics_dim=256, model_size_omic='small'):
        super(MultiSurvFix, self).__init__()
        self.img_att = Attn_Net_Gated(L=input_dim, D=hidden_dim)
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=omics_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        
        self.fusion = LRBilinearFusion(dim1=input_dim, dim2=hidden_dim, output_dim=output_dim)

    def forward(self, imgs, omics):
        # imgs N x F omics F
        A_path, h_path = self.img_att(imgs)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)

        h_omic = self.fc_omic(omics)
        return self.fusion(h_path, h_omic)
    

class MultiSurvConcat(nn.Module):
    def __init__(self, input_dim, output_dim=4, hidden_dim=256, omics_dim=256, model_size_omic='small'):
        super(MultiSurvConcat, self).__init__()
        self.img_att = Attn_Net_Gated(L=input_dim, D=hidden_dim)
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=omics_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        
        self.fusion = nn.Linear(hidden[-1] + input_dim, output_dim)

    def forward(self, imgs, omics):
        # imgs N x F omics F
        A_path, h_path = self.img_att(imgs)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)

        h_omic = self.fc_omic(omics)
        return self.fusion(torch.concatenate([h_path, h_omic], dim=-1))


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss

# def nll_loss(hazards, Y, c, S=None, alpha=0.4, eps=1e-8):
#   batch_size = len(Y)
#   Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
#   c = c.view(batch_size, 1).float() #censorship status, 0 or 1
#   if S is None:
#       S = 1 - torch.cumsum(hazards, dim=1) # surival is cumulative product of 1 - hazards
#   uncensored_loss = -(1 - c) * (torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
#   censored_loss = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps))
#   loss = censored_loss + uncensored_loss
#   loss = loss.mean()
#   return loss

class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None): 
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)

# loss_fn(hazards=hazards, S=S, Y=Y_hat, c=c, alpha=0)
class NLLSurvLoss_dep(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)

def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

class SNN(nn.Module):
    def __init__(self, omic_input_dim: int, model_size_omic: str='small', n_classes: int=4):
        super(SNN, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)


    def forward(self, x):
        h_omic = self.fc_omic(x)
        h  = self.classifier(h_omic) # logits needs to be a [B x 4] vector      
        assert len(h.shape) == 2 and h.shape[1] == self.n_classes
        return h

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')
        else:
            self.fc_omic = self.fc_omic.to(device)


        self.classifier = self.classifier.to(device)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class AMIL(nn.Module):
    def __init__(self, size_arg = "small", n_classes=4):
        super(AMIL, self).__init__()
        self.size_dict = {"small": [512, 512, 256], "big": [512, 512, 384]}
        size = self.size_dict[size_arg]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        self.classifier = nn.Linear(size[1], n_classes)
        initialize_weights(self)
                
                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
        else:
            self.attention_net = self.attention_net.to(device)

        self.classifier = self.classifier.to(device)


    def forward(self, x):

        A, x = self.attention_net(x)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=2)
        M = torch.bmm(A, x)
        h  = self.classifier(M)
        return h

    def get_slide_features(self, **kwargs):
        h = kwargs['x_path']

        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0)

        if 'attention_only' in kwargs.keys():
            if kwargs['attention_only']:
                return A

        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h) 
        return M


if __name__ == '__main__':
    # _input_omics = torch.ones((1, 128)).cuda()
    # _input_imgs = torch.ones((1, 128)).cuda()
    # model = LRBilinearFusion().to('cuda')
    # _output = model(_input_omics, _input_imgs)
    # print(_output.shape)

    # _input_path = torch.ones((32, 128))
    # model = Attn_Net_Gated(L=128, D=128)
    # A_path, h_path = model(_input_path)
    # A_path = torch.transpose(A_path, 1, 0)
    # h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
    # print(h_path.shape)

    _input_omics = torch.ones((1, 256)).cuda()
    _input_imgs = torch.ones((32, 256)).cuda()
    model = MultiSurv(input_dim=256).cuda()
    _output = model(_input_imgs, _input_omics)
    print(_output.shape)