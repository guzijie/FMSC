
import torch
import torch.nn as nn
import torch.nn.functional as F
EPS=1e-8


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        
    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)
    #MCE还得记知识点 有点怕跑通了就不记知识点了


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak) 
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold 
        if not (mask != 0).any():
            return 0
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None
        
        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        #+ margin_loss(anchors_weak, anchors_strong, neighbood = 20)
        
        return loss


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)  # 在第一个维度上做softmax 而不是第0个维度上
        self.bce = nn.BCELoss() # 二分类交叉熵损失
        self.entropy_weight = entropy_weight # Default = 2.0
        self.ce = nn.CrossEntropyLoss()  # 交叉熵
        self.t = 0.5
        self.confidence_ce = ConfidenceBasedCE(0.95, True)

    def forward(self, anchors, neighbors, anchor_augmented, clustering_results, index, batch_index):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        ta = 1
        tb = 1
        # Softmax #criterion(anchors_output_subhead, neighbors_output_subhead, anchor_augmented_output_subhead, clustering_results_head, index, initial_rank_index)
        b, n = anchors.size()
        # print('b,n anchors.size()',b,'\n n:',n) # 400,20 或者原来batchsize 500,20
        anchors_prob = self.softmax(anchors) # anchors softmax以后的概率值
        positives_prob = self.softmax(neighbors) # 是每个batch一个neighbor吗 这仨都通过softmax一下
        anchor_augmented_prob = self.softmax(anchor_augmented)

        positives_prob_augmented = anchors_prob[batch_index,:]

        positives_prob = torch.cat([positives_prob, positives_prob_augmented], dim=0)
        
        anchors_prob = torch.cat([anchors_prob, anchor_augmented_prob], dim=0)


        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(2 * b, 1, n), positives_prob.view(2 * b, n, 1)).squeeze() #b,n是anchor.size()的大小
        # print('similarity:',similarity)
        # print('similarity shape',similarity.shape)

        ones = torch.ones_like(similarity) # torch.ones_like返回一个填充了标量值1的张量，其大小与之相同 input

        consistency_loss = ta*tb*self.bce(similarity, ones)  # L_ins_local这个consistency_loss 是拿 similarity 和 ones 进行做 二分类交叉熵的

        kmeans_loss = 0
        if clustering_results is not None: # 全是global范围内的=L_ins_golbal + L_clas_global

            samples = torch.cat([anchors_prob, positives_prob], dim=0)

            
            anchors_prob_neighbood = clustering_results[index, :]
            
            anchors_prob_neighbood = anchors_prob_neighbood.repeat(4,1)

            similarity = torch.bmm(samples.view(4 * b, 1, n), anchors_prob_neighbood.view(4 * b, n, 1)).squeeze()


            ones = torch.ones_like(similarity)


            samples_c = F.normalize(samples, dim = 0)
            anchors_prob_neighbood_c = F.normalize(anchors_prob_neighbood, dim = 0)

            similarity_c = torch.mm(samples_c.t(), anchors_prob_neighbood_c)

            labels_c = torch.tensor(list(range(n))).cuda()

            kmeans_loss += ta*self.bce(similarity, ones) + self.ce(similarity_c, labels_c) # L_ins_golbal + L_clas_global都是和单位矩阵相减的感觉 那这kmeans不就是bce和ce两个加起来的


        
        anchors_prob_c = F.normalize(anchors_prob, dim = 0) # 意义感觉跟归一化差不多 将某一个维度除以那个维度对应的范数(默认是2范数) dim:0表示按列操作，则每列都是除以该列下平方和的开方；
        positives_prob_c = F.normalize(positives_prob, dim = 0)

        

        similarity = torch.mm(anchors_prob_c.t(), positives_prob_c) # 矩阵相乘有torch.mm和torch.matmul两个函数。其中前一个是针对二维矩阵，后一个是高维



        labels = torch.tensor(list(range(n))).cuda()



        ce_loss =  tb*self.ce(similarity, labels)
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)



        # Total loss
        total_loss = consistency_loss + ce_loss + kmeans_loss  - self.entropy_weight * entropy_loss #和文章里对应起来 consistency_loss包括了local和global的，那ce_loss应该对应了class损失的local和global的
        
        return total_loss, consistency_loss, ce_loss, entropy_loss



class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast #对比头的损失
    def __init__(self, temperature):  #传进来temperture  就是损失函数里的那个下面坟墓类似t的东西
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]  dim是128维的特征 b是什么

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss
