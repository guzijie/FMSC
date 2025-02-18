import argparse
import os
import torch
import faiss
import numpy as np
import random
import torch.nn.functional as F
import open_clip
from termcolor import colored
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations, get_train_dataset, \
    get_train_dataloader, get_val_dataset, get_val_dataloader, get_optimizer, get_model, get_criterion, \
    adjust_learning_rate
from utils.evaluate_utils import get_predictions, scan_evaluate, hungarian_evaluate
from utils.train_utils import scan_train
from utils.faiss_utils import search_index_pytorch, search_raw_array_pytorch
import time

FLAGS = argparse.ArgumentParser(description='SCAN Loss with CLIP Integration')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')
FLAGS.add_argument('--gpus', default='', type=str, help='available gpu list, leave empty to use cpu')
FLAGS.add_argument('--seed', default=None, type=int, help='random seed')


def main():
    args = FLAGS.parse_args()
    args.config_env = 'configs/env.yml'
    args.config_exp = 'configs/scan/scan_cifar10.yml'
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # Set CUDNN benchmark
    torch.backends.cudnn.benchmark = True

    # Load the CLIP model
    print(colored('Load CLIP model', 'blue'))
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.cuda().eval()

    # Get dataset and dataloaders
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)
    train_dataset = get_train_dataset(p, train_transformations, split='train', to_neighbors_dataset=True)
    val_dataset = get_val_dataset(p, val_transformations, to_neighbors_dataset=True)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print(f'Train samples: {len(train_dataset)} - Val samples: {len(val_dataset)}')

    # Fix random seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        print(f'Random seed fixed to {args.seed}')

    # Load the original model
    print(colored('Get original model', 'blue'))
    model = get_model(p, p['pretext_model'])
    model = model.cuda()

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])
    print(optimizer)

    # Loss function
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda()
    print(criterion)

    # Main loop
    print(colored('Starting main loop', 'blue'))
    for epoch in range(p['epochs']):
        print(colored(f'Epoch {epoch + 1}/{p["epochs"]}', 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust learning rate
        lr = adjust_learning_rate(p, optimizer, epoch)
        print(f'Adjusted learning rate to {lr:.5f}')

        # Training step with CLIP integration
        train_with_clip(train_dataloader, model, clip_model, criterion, optimizer, epoch)

        # Evaluate on validation set
        print('Evaluate on validation set ...')
        predictions, features = get_predictions(p, val_dataloader, model, return_features=True)
        scan_stats = scan_evaluate(predictions)
        print(scan_stats)
        lowest_loss_head = scan_stats['lowest_loss_head']
        lowest_loss = scan_stats['lowest_loss']

        clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
        print(clustering_stats)


def train_with_clip(train_dataloader, model, clip_model, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (batch_data, labels) in enumerate(train_dataloader):
        images = batch_data['anchor_augmented']
        images = images.cuda()
        labels = labels.cuda()

        # Forward pass through the original model
        outputs, features = model(images)  # 确保模型返回两个值

        # Extract features using CLIP
        with torch.no_grad():
            clip_features = clip_model.encode_image(images)

        # Normalize features and compute cosine similarity
        clip_features = F.normalize(clip_features, p=2, dim=1)
        model_features = F.normalize(features, p=2, dim=1)

        # 检查形状，确保它们可以进行矩阵乘法
        print(f"clip_features shape: {clip_features.shape}")
        print(f"model_features shape: {model_features.shape}")

        sim_matrix_clip = torch.mm(clip_features, clip_features.t())
        sim_matrix_model = torch.mm(model_features, model_features.t())

        # Calculate similarity loss
        sim_loss = F.mse_loss(sim_matrix_clip, sim_matrix_model)

        # Calculate total loss (including the original criterion loss)
        loss = criterion(outputs, labels) + sim_loss

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(f"Epoch [{epoch + 1}], Batch [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Sim Loss: {sim_loss.item():.4f}")


if __name__ == "__main__":
    main()
