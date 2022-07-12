
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity,LpDistance
from pytorch_metric_learning.regularizers import LpRegularizer

from torch.nn.parameter import Parameter
from torch.nn import init

import math

class SoftTriple(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K):
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SalLoss(nn.Module):
    def __init__(self):
        super(SalLoss, self).__init__()
        self.eps = 1e-6

    def KL_loss(self, input, target):
        input = input / input.sum() 
        target = target / target.sum()
        loss = (target * torch.log(target/(input+self.eps) + self.eps)).sum()
        return loss 

    def CC_loss(self, input, target):
        input = (input - input.mean()) / input.std()  
        target = (target - target.mean()) / target.std()
        loss = (input * target).sum() / (torch.sqrt((input*input).sum() * (target * target).sum()))
        loss = 1 - loss
        return loss

    def NSS_loss(self, input, target):
        ref = (target - target.mean()) / target.std()
        input = (input - input.mean()) / input.std()
        loss = (ref*target - input*target).sum() / target.sum()
        return loss 

    def forward(self, input, smap, fix):
        kl = 0
        cc = 0
        nss = 0
        for p, f, s in zip(input, fix, smap):
            kl += 1.0*self.KL_loss(p, s)
            cc += 0.5*self.CC_loss(p, s)
            nss += 0.2*self.NSS_loss(p, f)
        return (kl + cc + nss) / input.size(0)

class CLSLoss(nn.Module):
    """docstring for PYRMSELoss"""
    def __init__(self):
        super(CLSLoss, self).__init__()
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion=nn.BCEWithLogitsLoss()
        #self.criterion_m_mp = nn.CrossEntropyLoss(weight=self.weights_mp)
        #self.criterion_kl=nn.KLDivLoss(size_average=True, reduce=True)
        
    def forward(self,output,target):
        loss=0
        #loss += self.criterion(output, target.squeeze())
        loss += self.criterion(output, target)
        
        return loss

class TripletLoss(nn.Module):
	def __init__(self):
	    super(TripletLoss,self).__init__()
	    self.criterion=nn.BCEWithLogitsLoss()
	    self.miner=miners.MultiSimilarityMiner()
	    self.triplet_loss=losses.TripletMarginLoss()

	    
	def centerloss(self,out_1):

	    return ((out_1 ** 2).sum(dim=1).mean())  
	
	def forward(self,out_1,center,out_p_1,target):
	    
	    out_1 = out_1 - center

	    loss=0
	    target_mask=(1-target).bool()
	    out_1_masked=out_1.masked_select(target_mask).view(-1,out_1.size(1))
	    loss+=self.centerloss(out_1_masked)
	    #loss+=self.CLLoss(out_1,out_2)
	    
	    hard_pairs_1=self.miner(out_1,target.squeeze())
	    loss += self.triplet_loss(out_1,target.squeeze(),hard_pairs_1)
	    
	    loss += self.criterion(out_p_1, target)
	    
	    return loss

class TripletLosswithMiner(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    
    def __init__(self, margin,global_feat, labels):
        super(TripletLosswithMiner, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
 
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)	# batch_size
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class ContrastiveLoss(nn.Module):
	def __init__(self):
	    super(ContrastiveLoss,self).__init__()
	    self.criterion=nn.BCEWithLogitsLoss()
	    self.miner=miners.MultiSimilarityMiner()

	    self.triplet_loss=losses.TripletMarginLoss()

	    self.supconloss=SupConLoss(temperature=0.25, base_temperature=0.25)

	def CLLoss(self,out_1, out_2):
	    out_1 = F.normalize(out_1, dim=-1)
	    out_2 = F.normalize(out_2, dim=-1)
	    bs = out_1.size(0)
	    temp = 0.25
	    # [2*B, D]
	    out = torch.cat([out_1, out_2], dim=0)
	    # [2*B, 2*B]
	    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
	    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
	    # [2B, 2B-1]
	    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

	    # compute loss
	    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
	    # [2*B]
	    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
	    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
	    return loss
	
	    
	def centerloss(self,out_1,out_2):

	    return ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())  
	
	def centerloss_cos(self,out_n,out_a,center):
	    return (1-torch.mm(out_n,center.view(-1,1))).mean()#+torch.mm(out_a,center.unsqueeze(dim=1)).mean()
	
	def forward(self,out_1,out_2,center,out_p_1,out_p_2,target):
	    
	    loss=0
	    
	    
	    out_1 = out_1 - center
	    out_2 = out_2 - center
	    
	    #center loss
	    target_mask=(1-target).bool()
	    out_1_masked=out_1.masked_select(target_mask).view(-1,out_1.size(1))
	    out_2_masked=out_2.masked_select(target_mask).view(-1,out_2.size(1))
	    loss+=self.centerloss(out_1_masked,out_2_masked)
	    

	    #contrastive loss
	    #loss+=self.CLLoss(out_1,out_2)
	    out_1_norm = F.normalize(out_1, dim=-1)
	    out_2_norm = F.normalize(out_2, dim=-1)
	    features=torch.cat([out_1_norm.unsqueeze(1),out_2_norm.unsqueeze(1)],dim=1)
	    loss += self.supconloss(features,(1-target).view(-1,1))
	    
	    
	    #bce loss
	    loss += self.criterion(out_p_1, target)
	    loss += self.criterion(out_p_2, target)
	    
	    
	    return loss


class JOINTSALLoss(nn.Module):
    """docstring for PYRMSELoss"""
    def __init__(self):
        super(JOINTSALLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.criterion_m = nn.CrossEntropyLoss()
        self.kl_div=nn.KLDivLoss()
        self.cc=MyCorrCoef()
        
    def forward(self,output,target,output_m,target_m,output_sal,target_sal):
        loss=0
        
        loss += 0.5*self.criterion(output,target)
        loss += self.criterion_m(output_m, target_m)
        
        loss += 1.0 * self.kl_div(output_sal,target_sal)
        loss += 0.2 * self.cc(output_sal,target_sal)
        
        return loss


class SalMSELoss(nn.Module):
    def __init__(self):
        super(SalMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        #self.kl_div=KL_divergence()
        self.kl_div=nn.KLDivLoss()
        self.cc=MyCorrCoef()
        self.nss=NSS()
        #self.bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5]))
    
    def KL_loss(self, input, target):
        input = input / input.sum() 
        target = target / target.sum()
        loss = (target * torch.log(target/(input+self.eps) + self.eps)).mean()
        return loss 

    def forward(self, output, target_map,target_fix):
        
        loss=0
        #target_map=torch.unsqueeze(target_map,dim=1)
        loss += 10.0 * self.kl_div(output,target_map)
        #loss += 5.0 * self.criterion(output, target_map)
        #loss += 1.0 * self.bce(output[:,2,:,:].squeeze(),target_fix.squeeze())
        loss += 2.0 * self.cc(output,target_map)
        loss += 1.0 * self.nss(output,target_fix)
        

        return loss

class SalKLCCLoss(nn.Module):
    def __init__(self):
        super(SalKLCCLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        #self.kl_div=KL_divergence()
        self.kl_div=nn.KLDivLoss()
        self.cc=MyCorrCoef()
        self.eps=1e-10
    
    def KL_loss(self, input, target):
        input = input / input.sum() 
        target = target / target.sum()
        loss = (target * torch.log(target/(input+self.eps) + self.eps)).mean()
        return loss 

    def forward(self, output, target_map):
        
        loss=0
        output_norm=output/ (torch.sum(output,(2,3),keepdim=True) + self.eps)
        target_norm=target_map/ (torch.sum(target_map,(2,3),keepdim=True) + self.eps)
        
        loss += 5.0 * self.kl_div(output_norm.log(),target_norm)
        #loss += 5.0 * self.kl_div(output,target_map)
        loss += 1.0 * self.cc(output,target_map)
        

        return loss

class SalGraphLoss(nn.Module):
    def __init__(self):
        super(SalGraphLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.kl_div_1=nn.KLDivLoss()
        self.kl_div_2=nn.KLDivLoss()
        self.cc=MyCorrCoef()
        self.bce=nn.BCELoss()
        self.sigmoid=nn.Sigmoid()
        self.eps=1e-10

    def forward(self, output, target_map,output_node, target_node):
        loss=0
        #target_map=torch.unsqueeze(target_map,dim=1)
        output_norm=output/ (torch.sum(output,(2,3),keepdim=True) + self.eps)
        target_norm=target_map/ (torch.sum(target_map,(2,3),keepdim=True) + self.eps)
        
        loss += 5.0 * self.kl_div_1(output_norm.log(),target_norm)
        loss += 1.0 * self.cc(output,target_map)
        
        loss += 2.0 * self.kl_div_2(output_node,target_node)

        return loss




# KL-Divergence Loss
class KL_divergence(nn.Module):
    def __init__(self):
       super(KL_divergence,self).__init__()
       self.eps=1e-10
       self.kl_div=nn.KLDivLoss()
       
    def forward(self,y_pred_org,y_true_org):
        y_pred=torch.squeeze(y_pred_org)
        y_true=torch.squeeze(y_true_org)
        max_y_pred=torch.max(y_pred,2,keepdim=True)[0]
        max_y_pred=torch.max(max_y_pred,1,keepdim=True)[0]
        
        min_y_pred=torch.min(y_pred,2,keepdim=True)[0]
        min_y_pred=torch.min(min_y_pred,1,keepdim=True)[0]
        
        y_pred -= min_y_pred
        y_pred /= (max_y_pred-min_y_pred)
        
        
        sum_y_true=torch.sum(y_true,(1,2),keepdim=True)
        sum_y_pred=torch.sum(y_pred,(1,2),keepdim=True)
        
        y_true /= (sum_y_true + self.eps)
        y_pred /= (sum_y_pred + self.eps)
    
        return torch.mean(torch.sum(y_true * torch.log((y_true / (y_pred + self.eps)) + self.eps), (1,2)))
        #return self.kl_div(y_pred.log(),y_true)

class MyCorrCoef(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MyCorrCoef, self).__init__()

    def forward(self, pred, target):
        '''
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)

        CC = []

        for i in range(input.size(0)):
            im = input[i] - torch.mean(input[i])
            tm = target[i] - torch.mean(target[i])

            CC.append(-1.0*torch.sum(im * tm) / (torch.sqrt(torch.sum(im ** 2))
                                            * torch.sqrt(torch.sum(tm ** 2))))
            CC[i].unsqueeze_(0)

        CC = torch.cat(CC,0)
        CC = torch.mean(CC)

        return CC
        '''
        size = pred.size()
        new_size = (-1, size[-1] * size[-2])
        pred = pred.reshape(new_size)
        target = target.reshape(new_size)
    
        cc = []
        for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
            xm, ym = x - x.mean(), y - y.mean()
            r_num = torch.mean(xm * ym)
            r_den = torch.sqrt(
                torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
            r = -1.0*r_num / r_den
            cc.append(r)
    
        cc = torch.stack(cc)
        cc = cc.reshape(size[:2]).mean()
        return cc  # 1 - torch.square(r)

# Correlation Coefficient Loss
class Correlation_coefficient(nn.Module):
    def __init__(self):
        super(Correlation_coefficient,self).__init__()
        self.eps=1e-10
        
    def forward(self,y_pred,y_true):
        y_pred=torch.squeeze(y_pred)
        y_true=torch.squeeze(y_true)
        
        shape_r_out=y_pred.shape[1]
        shape_c_out=y_pred.shape[2]
        
        max_y_pred=torch.max(y_pred,2,keepdim=True)[0]
        max_y_pred=torch.max(max_y_pred,1,keepdim=True)[0]
        
        min_y_pred=torch.min(y_pred,2,keepdim=True)[0]
        min_y_pred=torch.min(min_y_pred,1,keepdim=True)[0]
        
        y_pred = y_pred-min_y_pred
        y_pred = y_pred/(max_y_pred-min_y_pred)
        
        
        sum_y_true=torch.sum(y_true,(1,2),keepdim=True)
        sum_y_pred=torch.sum(y_pred,(1,2),keepdim=True)
        
        y_true = y_true/(sum_y_true + self.eps)
        y_pred = y_pred/(sum_y_pred + self.eps)
    
        N = shape_r_out * shape_c_out
        sum_prod = torch.sum(y_true * y_pred,(1,2),keepdim=True)
        
        sum_x = torch.sum(y_true,(1,2),keepdim=True)
        sum_y = torch.sum(y_pred,(1,2),keepdim=True)
        
        sum_x_square = torch.sum(torch.square(y_true),(1,2),keepdim=True)
        sum_y_square = torch.sum(torch.square(y_pred),(1,2),keepdim=True)
    
        num = sum_prod - ((sum_x * sum_y) / N)
        
        den = torch.sqrt((sum_x_square - torch.square(sum_x) / N) * (sum_y_square - torch.square(sum_y) / N))
    
        return -1.0 * torch.mean(num / den)


# Normalized Scanpath Saliency Loss
class NSS(nn.Module):
    def __init__(self):
        super(NSS,self).__init__()
        self.eps=1e-10
        #self.flatten=nn.Flatten()
    
    def forward(self,pred,fixations):
        
        '''
        y_pred=torch.squeeze(y_pred)
        y_true=torch.squeeze(y_true)
        
        max_y_pred=torch.max(y_pred,2,keepdim=True)[0]
        max_y_pred=torch.max(max_y_pred,1,keepdim=True)[0]
        
        min_y_pred=torch.min(y_pred,2,keepdim=True)[0]
        min_y_pred=torch.min(min_y_pred,1,keepdim=True)[0]
        
        y_pred = y_pred-min_y_pred
        y_pred = y_pred/(max_y_pred-min_y_pred)
        
        
        #y_pred_flatten = self.flatten(y_pred)
    
        y_mean = torch.mean(y_pred, (1,2), keepdim=True)
        y_std = torch.mean(y_pred, (1,2), keepdim=True)

    
        y_pred = (y_pred - y_mean) / (y_std + self.eps)
    
        return -1.0*torch.mean(torch.sum(y_true * y_pred, (1,2)) / torch.sum(y_true, (1,2)))
        '''
        size = pred.size()
        new_size = (-1, size[-1] * size[-2])
        pred = pred.reshape(new_size)
        fixations = fixations.reshape(new_size)
    
        pred_normed = (pred - pred.mean(-1, True)) / pred.std(-1, keepdim=True)
        results = []
        for this_pred_normed, mask in zip(torch.unbind(pred_normed, 0),
                                          torch.unbind(fixations, 0)):
            if mask.sum() == 0:
                print("No fixations.")
                results.append(torch.ones([]).float().to(fixations.device))
                continue
            
            nss_ = torch.masked_select(this_pred_normed, mask)
            nss_ = nss_.mean(-1)
            results.append(-1.0*nss_)
        results = torch.stack(results)
        results = results.reshape(size[:2]).mean()
        #results = torch.mean(results)
        
        return results

class MyNormScanSali(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MyNormScanSali, self).__init__()

    def forward(self, input, target):

        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)

        NSS = []
        target_logic = torch.zeros(input.size(0), input.size(1) ).cuda()

        for i in range(input.size(0)):

            # normalize the predicted maps
            input_norm = (input[i] - torch.mean(input[i])) / torch.std(input[i])

            # compute the logic matrix of fixs
            for m in range(input.size(1)):
                if target[i,m] != 0:
                    target_logic[i,m] = 1

            NSS.append(torch.mean(-1.0*torch.mul(input_norm, target_logic[i])))
            NSS[i].unsqueeze_(0)

        NSS = torch.cat(NSS, 0)
        NSS = torch.mean(NSS)

        return NSS
    


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
