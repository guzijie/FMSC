"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import faiss
import numpy as np
import random
import os
import torch.nn.functional as F

from termcolor import colored
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_optimizer, get_model, get_criterion,\
                                adjust_learning_rate
from utils.evaluate_utils import get_predictions, scan_evaluate, hungarian_evaluate
from utils.train_utils import scan_train

from utils.faiss_utils import search_index_pytorch, search_raw_array_pytorch
import numpy as np
import time
import open_clip

FLAGS = argparse.ArgumentParser(description='SCAN Loss')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')
FLAGS.add_argument('--gpus', default='', type=str,
                            help='available gpu list, leave empty to use cpu')
FLAGS.add_argument('--seed', default=None, type=int,
                            help='random seed')

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def main():
    args = FLAGS.parse_args()
    args.config_env = 'configs/env.yml'
    args.config_exp = 'configs/scan/scan_stl10.yml'
    p = create_config(args.config_env, args.config_exp)  # python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml
    print(colored(p, 'red'))

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)
    train_dataset = get_train_dataset(p, train_transformations,
                                        split='train', to_neighbors_dataset = True)
    val_dataset = get_val_dataset(p, val_transformations, to_neighbors_dataset = True)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train transforms:', train_transformations)
    print('Validation transforms:', val_transformations)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    # clip
    model_clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='vit_b_32-laion2b_e16-af8dbd0c.pth')
    clip_model = model_clip.cuda().eval()
    # fix random seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        print('Random seed will be fixed to %d' % args.seed)

    #torch.cuda.set_device('cuda:2')

    # Model   model = ClusteringModel(backbone, p['num_classes'], p['num_heads'])
    print(colored('Get model', 'blue')) # cifar数据集的用backbone =resnet18() ，p['setup'] in ['scan', 'selflabel']: model = ClusteringModel(backbone, p['num_classes'], p['num_heads'])
    model = get_model(p, p['pretext_model']) # 是simclr对比学习头 预训练的模型cfg['pretext_model'] =./results/cifar-10/pretext/model.pth.tar
    # print(model) # setup: scan  #def get_model(p, pretrain_path=None):，p['pretext_model']是None
    # data parallel
    if len(args.gpus.split(',')) >= 1:
        print('Data parallel will be used for acceleration purpose')
        device_ids = [int(x) for x in args.gpus.split(',')]
        torch.cuda.set_device(f'cuda:{device_ids[0]}')
        model = torch.nn.DataParallel(model, device_ids)

    model = model.cuda()

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])
    print(optimizer)

    # Warning
    if p['update_cluster_head_only']:
        print(colored('WARNING: SCAN will only update the cluster head', 'red'))

    # Loss function
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)  # 这句就是criterion = SCANLoss(entropy_weight=5.0) SCANLoss在losses.py里
    criterion.cuda()
    print(criterion)

    clustering_results = None

    # Checkpoint
    if os.path.exists(p['scan_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['scan_checkpoint']), 'blue'))
        checkpoint = torch.load(p['scan_checkpoint'])
        model.load_state_dict(checkpoint['model']) # 会在这里改变load进来的model吗？model = get_model(p, p['pretext_model'])之前写的是simclr
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_head = checkpoint['best_loss_head']
        clustering_results = checkpoint['clustering_results']
        best_acc = checkpoint['best_acc']


    else:
        print(colored('No checkpoint file at {}'.format(p['scan_checkpoint']), 'blue'))
        start_epoch = 0
        best_loss = 1e4
        best_loss_head = None
        best_acc = 0.




    # Main loop
    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train 看清这里的model，是对比学习simclr的model，在训练提特征？
        print('Train ...') # 这里传进来的 criterion 就是 loss 函数SCANLoss 函数，scan_train返回的是 total_loss, consistency_loss, ce_loss, entropy_loss
        ## 训练模型
        scan_train(train_dataloader, model, criterion, optimizer, epoch, p['update_cluster_head_only'], clustering_results, clip_model, preprocess_train)

        # Evaluate
        print('Obtain prediction on train set ...') # 模型验证时候的eval 和最终测试模型效果时候的eval都是同一个函数
        out, features = get_predictions(p, train_dataloader, model, return_features=True)

        print('Execute nn_serach ...')
        with torch.no_grad():
            clustering_results = nn_serach(features, out, p) # 在特征层面上找最近邻 所以是在scan_train 这个model moco网络输出最近邻以后，

        # Evaluate
        print('Make prediction on validation set ...')
        predictions = get_predictions(p, val_dataloader, model)

        print('Evaluate based on SCAN loss ...')
        scan_stats = scan_evaluate(predictions)
        print(scan_stats)
        lowest_loss_head = scan_stats['lowest_loss_head']
        lowest_loss = scan_stats['lowest_loss']

        print('Evaluate with hungarian matching algorithm ...')
        clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
        print(clustering_stats)

        if best_acc < clustering_stats['ACC']:
            print('New lowest loss on validation set: %.4f -> %.4f' %(best_loss, lowest_loss))
            print('Best ACC on validation set: %.4f -> %.4f' %(best_acc, clustering_stats['ACC']))
            print('Lowest loss head is %d' %(lowest_loss_head))
            best_loss = lowest_loss
            best_loss_head = lowest_loss_head
            best_acc = clustering_stats['ACC']
            torch.save({'model': model.module.state_dict(), 'head': best_loss_head}, p['scan_model'])

        else:
            print('No new lowest loss on validation set: %.4f -> %.4f' %(best_loss, lowest_loss))
            print('Lowest loss head is %d' %(best_loss_head))

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1, 'best_loss': best_loss, 'best_acc': best_acc, 'best_loss_head': best_loss_head, 'clustering_results': clustering_results},
                     p['scan_checkpoint'])

    # Evaluate and save the final model
    print(colored('Evaluate best model based on SCAN metric at the end', 'blue'))
    model_checkpoint = torch.load(p['scan_model'], map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])
    predictions = get_predictions(p, val_dataloader, model)
    clustering_stats = hungarian_evaluate(model_checkpoint['head'], predictions,
                            class_names=val_dataset.dataset.classes,
                            compute_confusion_matrix=True,
                            confusion_matrix_file=os.path.join(p['scan_dir'], 'confusion_matrix.png')) # 跑过scan.py以后得到 就是输出return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}
    print(clustering_stats)


def nn_serach(x, out, p):
    """
    Args:
        x: features to be clustered
        out: prediction_out
    """

    im2cluster = []

    search_neighbors = 2
    features = x.to('cuda')  # 确保 features 在 GPU 上
    start = time.time()

    # GPU + PyTorch CUDA Tensors (1)
    res = faiss.StandardGpuResources()  # 调用faiss找最近邻
    res.setDefaultNullStreamAllDevices()
    _, initial_rank = search_raw_array_pytorch(res, features, features, search_neighbors)
    end = time.time()
    print('the elapsed time is ', (end - start))

    if search_neighbors > 2:
        index = np.random.choice((1, search_neighbors - 1), 1)[0]
        initial_rank_index = initial_rank[:, index].squeeze()
    else:
        initial_rank_index = initial_rank[:, -1].squeeze()

    # 将 initial_rank_index 移动到 GPU 上
    initial_rank_index = initial_rank_index.to('cuda')

    for head in out:
        features = head['probabilities'].to('cuda')  # 确保 features 在 GPU 上

        # 使用 initial_rank_index 进行索引
        features = features[initial_rank_index, :]

        im2cluster.append(features)

    return im2cluster


if __name__ == "__main__":
    main()
