"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.faiss_utils import search_index_pytorch, search_raw_array_pytorch
import faiss
import diffusion
def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    diffusion.generate_samples_with_diffusion()
    for i, (batch, index) in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


from torchvision.transforms import ToPILImage

# 定义转换为PIL图像的转换
to_pil = ToPILImage()


def scan_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False, clustering_results=None, model_clip=None, preprocess_train=None):
    """
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    ce_losses = AverageMeter('Class Cross Entropy', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [total_losses, consistency_losses, ce_losses, entropy_losses],
                             prefix="Epoch: [{}]".format(epoch))

    res = faiss.StandardGpuResources()
    res.setDefaultNullStreamAllDevices()

    if update_cluster_head_only:
        model.eval()
    else:
        model.train()

    for i, (batch, index) in enumerate(train_loader):
        anchors = batch['anchor'].to('cuda', non_blocking=True)
        neighbors = batch['neighbor'].to('cuda', non_blocking=True)
        anchor_augmented = batch['anchor_augmented'].to('cuda', non_blocking=True)

        if update_cluster_head_only:
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
                anchor_augmented_features = model(anchor_augmented, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')
            anchor_augmented_output = model(anchor_augmented_features, forward_pass='head')
        else:
            anchors_features = model(anchors, forward_pass='backbone')
            neighbors_features = model(neighbors, forward_pass='backbone')
            anchor_augmented_features = model(anchor_augmented, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')
            anchor_augmented_output = model(anchor_augmented_features, forward_pass='head')

        # CLIP特征提取
        with torch.no_grad():
            anchors_clip_preprocessed = torch.stack([preprocess_train(to_pil(img.cpu())) for img in anchors])
            anchors_clip_preprocessed = anchors_clip_preprocessed.to('cuda')
            clip_features = model_clip.encode_image(anchors_clip_preprocessed)

        clip_norms = torch.linalg.norm(clip_features, dim=1)
        model_norms = torch.linalg.norm(anchors_features, dim=1)

        clip_sim_matrix = torch.mm(clip_features, clip_features.t()) / (clip_norms.unsqueeze(1) * clip_norms.unsqueeze(0))
        model_sim_matrix = torch.mm(anchors_features, anchors_features.t()) / (model_norms.unsqueeze(1) * model_norms.unsqueeze(0))

        sim_loss = F.mse_loss(clip_sim_matrix, model_sim_matrix)

        _, initial_rank = search_raw_array_pytorch(res, anchors_features, anchors_features, 2)
        initial_rank_index = initial_rank[:, -1].squeeze()

        total_loss, consistency_loss, ce_loss, entropy_loss = [], [], [], []
        for j, (anchors_output_subhead, neighbors_output_subhead, anchor_augmented_output_subhead) in enumerate(zip(anchors_output, neighbors_output, anchor_augmented_output)):
            if clustering_results is not None:
                clustering_results_head = clustering_results[j]
            else:
                clustering_results_head = None
            total_loss_, consistency_loss_, ce_loss_, entropy_loss_ = criterion(
                anchors_output_subhead, neighbors_output_subhead, anchor_augmented_output_subhead,
                clustering_results_head, index, initial_rank_index)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)
            ce_loss.append(ce_loss_)

        total_loss = torch.sum(torch.stack(total_loss, dim=0)) + 0.1 * sim_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)



def selflabel_train(train_loader, model, criterion, optimizer, epoch, ema=None):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, (batch, index) in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad(): 
            output = model(images)
        output_augmented = model(images_augmented)

        # print(output.shape)
        # print(output_augmented.shape)
        # Loss for every head
        loss = []
        for j, (anchors_output_subhead, neighbors_output_subhead) in enumerate(zip(output, output_augmented)):

            loss_ = criterion(anchors_output_subhead, neighbors_output_subhead)
            loss.append(loss_)

        losses.update(np.mean([v.item() for v in loss]))
        loss = torch.sum(torch.stack(loss, dim=0))
        
        # loss = criterion(output, output_augmented)
        # losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)
        
        if i % 25 == 0:
            progress.display(i)
