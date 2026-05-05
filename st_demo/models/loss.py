import torch
import torch.nn.functional as F
from torch import nn

def cosine_sim(x: torch.Tensor, y: torch.Tensor):
    """Cosine similarity between all the image and sentence pairs. Assumes that x and y are l2 normalized"""
    return x.bmm(y.transpose(-1, -2))

def l2norm(x):
    """L2-normalize columns of x"""
    return F.normalize(x, p=2, dim=-1)

def rbf(x, y, gamma):
    """RBF kernel K(x,y) """
    pdist = torch.norm(x[:, None] - y, dim=2, p=2)
    return torch.exp(-gamma * pdist)

def rbf_memory_efficient(x, y, gamma):
    """RBF kernel that does not cause memory shortage"""
    cdist = torch.cdist(x, y)
    return torch.exp(-gamma * cdist)

def Tripletloss(anchor, ps, ns, margin=1, distance_fn=cosine_sim):
    pos_dist = distance_fn(ps, anchor)
    neg_dist = distance_fn(ns, anchor)
    loss = (margin - pos_dist + neg_dist).clamp(min=0.0, max=margin)
    num_triplets = torch.nonzero(loss).shape[0]
    if num_triplets == 0:
        return loss.mean()
    else:
        return loss.sum() / num_triplets
    
def Cliploss(anchor, ps, ns, distance_fn=cosine_sim):
    pos_dist = distance_fn(ps, anchor)
    neg_dist = distance_fn(ns, anchor)
    pred_tensor = torch.concat([pos_dist, neg_dist], dim=-1).squeeze(0)
    gt_tensor = torch.zeros(1, dtype=torch.long).to(pred_tensor.device)
    loss = F.cross_entropy(pred_tensor, gt_tensor)
    return loss

def SetTripletLoss(img_embs, neg_embs, gene_embs, margin, distance_fn, dist_choice='cosine'):
    # Compute setwise distance with provided set distance metric
    pos_setwise_dist = distance_fn(img_embs, gene_embs, dist_choice)
    neg_setwise_dist = distance_fn(neg_embs, gene_embs, dist_choice)
    loss = (margin - pos_setwise_dist + neg_setwise_dist).clamp(min=0.0, max=margin)
    num_triplets = torch.nonzero(loss).shape[0]
    if num_triplets == 0:
        return loss.mean()
    else:
        return loss.sum() / num_triplets
    
def mmd_rbf_loss(x, y, gamma=None, reduction='mean'):
    if gamma is None:
        gamma = 1./x.size(-1)
    if reduction=='mean':
        loss = rbf_memory_efficient(x, x, gamma).mean() - 2 * rbf_memory_efficient(x, y, gamma).mean() + rbf_memory_efficient(y, y, gamma).mean()
    else:
        loss = rbf_memory_efficient(x, x, gamma).sum() - 2 * rbf_memory_efficient(x, y, gamma).sum() + rbf_memory_efficient(y, y, gamma).sum()
    return loss

def recon_loss(x, y):
    return torch.norm(x-y, p='fro') / (x.shape[0] * x.shape[1])

def smooth_chamfer_distance(img_embs, gene_embs, dist_fn='l2'):
        """
            cosine version of smooth_chamfer_distance_euclidean(img_embs, gene_embs)
            img_embs: BxNxF
            gene_embs: Bx1xF
        """
        assert len(img_embs.shape) == 3
        if dist_fn == 'cosine':
            dist = cosine_sim(img_embs, gene_embs)  # BxNx1
        else:
            dist = torch.cdist(img_embs, gene_embs)  # BxNx1
        img_set_size = img_embs.shape[1] * img_embs.shape[0]
        gene_set_size = gene_embs.shape[1] * gene_embs.shape[0]
        
        right_term = torch.sum(
            torch.log(torch.sum(
                torch.exp(dist), axis=-2, keepdim=True
            )), axis=-1, keepdim=True).squeeze()
        left_term = torch.sum(
            torch.log(torch.sum(
                torch.exp(dist), axis=-1, keepdim=True
            )), axis=-2, keepdim=True).squeeze()
        smooth_chamfer_dist = (right_term / gene_set_size + left_term / img_set_size) / 2

        return smooth_chamfer_dist

def omic_sim_loss(pos_omic, neg_omic, anchor_img, temperature=1):
    """
    pos_omic B 1 F
    neg_omic B 1 N F
    anchor_img B 1 F
    """
    pos_omic, neg_omic, anchor_img = \
        pos_omic.squeeze(1), neg_omic.squeeze(1), anchor_img.squeeze(1)
    pos_sim = F.cosine_similarity(pos_omic, anchor_img, dim=-1) / temperature
    anchor_img_expanded = anchor_img.unsqueeze(1)
    neg_sim = F.cosine_similarity(anchor_img_expanded, neg_omic, dim=-1) / temperature
    loss_value = -torch.log(
        torch.exp(pos_sim) / torch.sum(torch.exp(neg_sim))
    )
    return loss_value


class MPdistance(nn.Module):
    def __init__(self, avg_pool):
        super(MPdistance, self).__init__()
        self.avg_pool = avg_pool
        self.alpha, self.beta = nn.Parameter(torch.ones(1)).cuda(), nn.Parameter(torch.zeros(1)).cuda()
        
    def forward(self, img_embs, gene_embs):
        dist = cosine_sim(img_embs, gene_embs)
        avg_distance = self.avg_pool(torch.sigmoid(self.alpha * dist.unsqueeze(0) + self.beta)).squeeze(0)
        return avg_distance

class SetwiseDistance(nn.Module):
    def __init__(self, img_set_size, gene_set_size, denominator, temperature=1, temperature_gene_scale=1):
        super(SetwiseDistance, self).__init__()
        # poolings
        self.img_set_size = img_set_size
        self.gene_set_size = gene_set_size
        self.denominator = denominator
        self.temperature = temperature
        self.temperature_gene_scale = temperature_gene_scale # used when computing i2t distance
        
        self.xy_max_pool = torch.nn.MaxPool2d((self.img_set_size, self.gene_set_size))
        self.xy_avg_pool = torch.nn.AvgPool2d((self.img_set_size, self.gene_set_size))
        self.x_axis_max_pool = torch.nn.MaxPool2d((1, self.gene_set_size))
        self.x_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(1, self.gene_set_size))
        self.y_axis_max_pool = torch.nn.MaxPool2d((self.img_set_size, 1))
        self.y_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(self.img_set_size, 1))
        
        self.mp_dist = MPdistance(self.xy_avg_pool)
        
    def smooth_chamfer_distance_euclidean(self, img_embs, gene_embs):
        """
            Method to compute Smooth Chafer Distance(SCD). Max pool is changed to LSE.
            Use euclidean distance(L2-distance) to measure distance between elements.
        """
        dist = torch.cdist(img_embs, gene_embs)
        
        right_term = -self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(-self.temperature * self.temperature_gene_scale * dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = -self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(-self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = (right_term / (self.img_set_size * self.temperature * self.temperature_gene_scale) + left_term / (self.gene_set_size * self.temperature)) / (self.denominator)

        return smooth_chamfer_dist
    
    def smooth_chamfer_distance_cosine(self, img_embs, gene_embs):
        """
            cosine version of smooth_chamfer_distance_euclidean(img_embs, gene_embs)
        """
        dist = cosine_sim(img_embs, gene_embs)
        
        right_term = self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(self.temperature * self.temperature_gene_scale * dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = (right_term / (self.img_set_size * self.temperature * self.temperature_gene_scale) + left_term / (self.gene_set_size * self.temperature)) / (self.denominator)

        return smooth_chamfer_dist
    
    def chamfer_distance_cosine(self, img_embs, gene_embs):
        """
            cosine version of chamfer_distance_euclidean(img_embs, gene_embs)
        """
        dist = cosine_sim(img_embs, gene_embs)
        
        right_term = self.y_axis_sum_pool(self.x_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        left_term = self.x_axis_sum_pool(self.y_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        
        chamfer_dist = (right_term / self.img_set_size + left_term / self.gene_set_size) / self.denominator

        return chamfer_dist
    
    def max_distance_cosine(self, img_embs, gene_embs):
        dist = cosine_sim(img_embs, gene_embs)
        max_distance = self.xy_max_pool(dist.unsqueeze(0)).squeeze(0)
        return max_distance

    def smooth_chamfer_distance(self, img_embs, gene_embs):
        return self.smooth_chamfer_distance_cosine(img_embs, gene_embs)
    
    def chamfer_distance(self, img_embs, gene_embs):
        return self.chamfer_distance_cosine(img_embs, gene_embs)
    
    def max_distance(self, img_embs, gene_embs):
        return self.max_distance_cosine(img_embs, gene_embs)
    
    def avg_distance(self, img_embs, gene_embs):
        return self.mp_dist(img_embs, gene_embs)