# import os
# import errno
##pyrotch1.4版本
# dirpath='/data/wnx/'
# def tryosmakedir(dirpath):
#     try:
#         os.makedirs(dirpath)
#         print("chenggong")
#     except OSError as e:
#         if e.errno == errno.EEXIST:
#             print("error:",e.errno)
#             print("error e:",e)
#             pass
#         else:
#             print("makedir 失败")
#             raise
#
#
# import torch.nn as nn
# import torch.nn.functional as F
#
# class LeNet(nn.Module):#定义类
#     def __init__(self):#定义初始化
#         super(LeNet, self).__init__()#super（）继承父类的构造函数
#         #搭建网络
#         self.conv1 = nn.Conv2d(3, 16, 5)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, 5)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))    # input(3, 32, 32) N=（W-F+2P）/S+1 output(16, 28, 28)
#         x = self.pool1(x)            # output(16, 14, 14)
#         x = F.relu(self.conv2(x))    # output(32, 10, 10)
#         x = self.pool2(x)            # output(32, 5, 5)
#         x = x.view(-1, 32*5*5)       # output(32*5*5) 压平
#         x = F.relu(self.fc1(x))      # output(120)
#         x = F.relu(self.fc2(x))      # output(84)
#         x = self.fc3(x)              # output(10)
#         return x
# import torch
# import torchvision
# import torch.nn as nn
# # from model import LeNet
# import torch.optim as optim
# import torchvision.transforms as transforms
#
#
# def main():
#     transform = transforms.Compose(
#         [transforms.ToTensor(),                                   #将numpy数据转换为Tensor
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #标准化
#
#
#     # 50000张训练图片
#     # 第一次使用时要将download设置为True才会自动去下载数据集
#     train_set = torchvision.datasets.CIFAR10(root='./data_try', train=True,
#                                              download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,#一次36张图片
#                                                shuffle=True, num_workers=0)
#
#     # 10000张验证图片
#     # 第一次使用时要将download设置为True才会自动去下载数据集
#     val_set = torchvision.datasets.CIFAR10(root='./data_try', train=False,
#                                            download=True, transform=transform)
#     val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
#                                              shuffle=False, num_workers=0)
#     #获取数据
#     val_data_iter = iter(val_loader)
#     val_image, val_label = val_data_iter.next()
#
#     # classes = ('plane', 'car', 'bird', 'cat',
#     #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#     net = LeNet()
#     loss_function = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(net.parameters(), lr=0.001)#损失函数
#
#     for epoch in range(5):  # loop over the dataset multiple times
#
#         running_loss = 0.0
#         for step, data in enumerate(train_loader, start=0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#
#             # 清零历史梯度
#             optimizer.zero_grad()
#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = loss_function(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             # print statistics
#             running_loss += loss.item()
#             if step % 500 == 499:    # print every 500 mini-batches
#                 with torch.no_grad():
#                     outputs = net(val_image)  # [batch, 10]
#                     predict_y = torch.max(outputs, dim=1)[1]#寻找最大值
#                     accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
#                     #计算准确率
#                     print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
#                           (epoch + 1, step + 1, running_loss / 500, accuracy))
#                     running_loss = 0.0
#
#     print('Finished Training')
#
#     save_path = './Lenet.pth'
#     torch.save(net.state_dict(), save_path)
#
#
# if __name__ == '__main__':
#     main()
#
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
#
# # from model import LeNet
#
#
# def main():
#     transform = transforms.Compose(
#         [transforms.Resize((32, 32)),
#          transforms.ToTensor(),
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#     classes = ('plane', 'car', 'bird', 'cat',
#                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#     net = LeNet()
#     net.load_state_dict(torch.load('Lenet.pth'))
#
#     im = Image.open('1.jpg')
#     im = transform(im)  # [C, H, W]
#     im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
#
#     with torch.no_grad():
#         outputs = net(im)
#         predict = torch.max(outputs, dim=1)[1].data.numpy()
#     print(classes[int(predict)])
#
#
# if __name__ == '__main__':
#     main()

#=============================
# import os
# import sys
# import os.path
# import hashlib
# from torch.utils.data import Dataset
# from torchvision.datasets.utils import check_integrity, download_and_extract_archive
# def calculate_md5(fpath, chunk_size=1024 * 1024):
#     md5 = hashlib.md5()
#     with open(fpath, 'rb') as f:
#         for chunk in iter(lambda: f.read(chunk_size), b''):
#             md5.update(chunk)
#     return md5.hexdigest()
#
#
# def check_md5(fpath, md5, **kwargs):
#     return md5 == calculate_md5(fpath, **kwargs)
#
#
# def check_integrity(fpath, md5=None):
#     if not os.path.isfile(fpath):
#         return False
#     if md5 is None:
#         return True
#     return check_md5(fpath, md5)
#
# class CIFAR10(Dataset):  #from torch.utils.data import Dataset
#     base_folder = 'cifar-10-batches-py'
#     url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
#     filename = "cifar-10-python.tar.gz"
#     tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
#     train_list = [
#         ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
#         ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
#         ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
#         ['data_batch_4', '634d18415352ddfa80567beed471001a'],
#         ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
#     ]
#
#     test_list = [
#         ['test_batch', '40351d587109b95175f43aff81a1287e'],
#     ]
#     meta = {
#         'filename': 'batches.meta',
#         'key': 'label_names',
#         'md5': '5ff9c542aee3614f3951f8cda6e48888',
#     }
#     def _check_integrity(self):
#         root = self.root
#         for fentry in (self.train_list + self.test_list):
#             filename, md5 = fentry[0], fentry[1]
#             fpath = os.path.join(root, self.base_folder, filename)
#             if not check_integrity(fpath, md5):
#                 return False
#         return True
#     def download(self):
#         if self._check_integrity():
#             print('Files already downloaded and verified')
#             return
#         download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
#
# path='/data/wnx/cifar-10-batches-py/data_batch_1'
# statinfo=os.stat(path)
# print(statinfo)

import os
import sys
import os.path
import hashlib
from torch.utils.data import Dataset
import torch
import numpy as np
import collections
from torch._six import string_classes, int_classes
import torchvision
import torch.nn as nn
import torch.nn.functional as F

#train_data=torchvision.datasets.CIFAR10(root='data/wnx',train=True,transform=torchvision.transforms.ToTensor(),download=True)
def collate_custom(batch):
    if isinstance(batch[0], np.int64):
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], string_classes):
        return batch

    elif isinstance(batch[0], collections.Mapping):
        batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        return batch_modified

    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))
def get_val_dataset(p, transform=None, to_neighbors_dataset=False):
    # Base dataset
    if p['val_db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=False, transform=transform, download=True) #难道是因为dataset没有daownload下来？？？

    elif p['val_db_name'] == 'cifar-20':
        from data.cifar import CIFAR20
        dataset = CIFAR20(train=False, transform=transform, download=True)

    elif p['val_db_name'] == 'stl-10':
        from data.stl import STL10
        dataset = STL10(split='test', transform=transform, download=True)

    elif p['val_db_name'] == 'imagenet':
        from data.imagenet import ImageNet
        dataset = ImageNet(split='val', transform=transform)

    elif p['val_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200', 'imagenet_10', 'imagenet_dogs']:
        from data.imagenet import ImageNetSubset
        subset_file = './data/imagenet_subsets/%s.txt' %(p['val_db_name'])
        dataset = ImageNetSubset(subset_file=subset_file, split='val', transform=transform)

    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))

    # Wrap into other dataset (__getitem__ changes)
    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['topk_neighbors_val_path'])
        dataset = NeighborsDataset(dataset, indices, 5) # Only use 5

    return dataset
def get_val_dataloader(p, dataset):

    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
                                       batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                                       drop_last=False, shuffle=False)
import argparse
import yaml
import torchvision.transforms as transforms
def get_val_transformations(p):
    return transforms.Compose([
        transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(**p['transformation_kwargs']['normalize'])])
FLAGS = argparse.ArgumentParser(description='Evaluate models from the model zoo')
args = FLAGS.parse_args()
with open('configs/scan/scan_cifar10.yml', 'r') as stream:
    config = yaml.safe_load(stream)
config['batch_size'] = 512  # To make sure we can evaluate on a single 1080ti ,batchsize改成512了,原来scan_cifar10.yml里是200
print(config)
transforms = get_val_transformations(config)
dataset = get_val_dataset(config, transforms) # 应该不会拿错，都是get_val的函数
dataloader = get_val_dataloader(config, dataset)
print('dataset:',dataset)
print('dataloader:',dataloader)
# {'setup': 'scan', 'criterion': 'scan', 'criterion_kwargs': {'entropy_weight': 5.0}, 'update_cluster_head_only': False, 'num_heads': 3, 'backbone': 'resnet18', 'train_db_name': 'cifar-10', 'val_db_name': 'cifar-10', 'num_classes': 10, 'num_neighbors': 20, 'augmentation_strategy': 'ours', 'augmentation_kwargs': {'crop_size': 32, 'normalize': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.201]}, 'num_strong_augs': 4, 'cutout_kwargs': {'n_holes': 1, 'length': 16, 'random': True}}, 'transformation_kwargs': {'crop_size': 32, 'normalize': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.201]}}, 'optimizer': 'adam', 'optimizer_kwargs': {'lr': 0.0001, 'weight_decay': 0.0001}, 'epochs': 100, 'batch_size': 512, 'num_workers': 8, 'scheduler': 'constant'}
# Files already downloaded and verified
# dataset: <data.cifar.CIFAR10 object at 0x7fc5623eeb10>
# dataloader: <torch.utils.data.dataloader.DataLoader object at 0x7fc5623feb90>
print(dataloader.dataset.data)
# [[[[158 112  49]
#    [159 111  47]
#   [165 116  51]
#      ...
#  [137  95  36]
# [126  91  36]
# [116  85  33]]....
print(dataloader.dataset.data.shape) # (10000, 32, 32, 3)
i=0
# for data in dataloader:#弄整个的data还是只弄  batch,_ , _
#     #images = batch[key_].cuda(non_blocking=True)
#     #print(batch)
#     #print(batch['image'])
#     print(data)
#     print('data[image]',data['image']) # data[image] tensor([[[[ 6.3375e-01,  6.5314e-01,  7.6945e-01,  ...,
#     if i==0:
#         break;
for batch in dataloader:
    print('batch',batch)
    images = batch['image'].cuda(non_blocking=True) #这句不就是batch['image'].cuda(non_blocking=True)吗 TypeError: string indices must be integers 但这个key_就是'image'啊
    bs = images.shape[0]
    print('images',images)
    print('bs',bs) # bs 512
    #res = model(images, forward_pass='return_all')
    #output = res['output']
    if i==0:
        break;
    #print(batch)
#这个datat打印出来也是有'image'的，那为什么images = batch['image'].cuda(non_blocking=True)这句会出现string indices must be integers
# {'image': tensor([[[[ 6.3375e-01,  6.5314e-01,  7.6945e-01,  ...,  2.2667e-01,
#                       1.3434e-02, -1.8042e-01],
#                     [ 5.1744e-01,  4.9806e-01,  6.5314e-01,  ...,  2.0728e-01,
#                       -5.9512e-03, -1.2226e-01],
#                     [ 4.9806e-01,  4.9806e-01,  6.3375e-01,  ...,  2.6544e-01,
#                       9.0974e-02, -1.0288e-01],
#                     ...,...,
#                     [-1.1483e+00, -1.0313e+00, -9.9224e-01,  ..., -1.0313e+00,
#                      -1.3434e+00, -1.2069e+00],
#                     [-1.5776e+00, -1.3629e+00, -1.3434e+00,  ..., -7.5812e-01,
#                      -1.1483e+00, -8.5567e-01],
#                     [-1.1678e+00, -8.9469e-01, -1.1288e+00,  ..., -1.1678e+00,
#                      -9.5322e-01, -1.0118e+00]]]]), 'target': tensor([3, 8, 8, 0, 6, 6, 1, 6, 3, 1, 0, 9, 5, 7, 9, 8, 5, 7, 8, 6, 7, 0, 4, 9,
#                                                                       5, 2, 4, 0, 9, 6, 6, 5, 4, 5, 9, 2, 4, 1, 9, 5, 4, 69, 9, 4, 5, 6])
#     , 'meta': {'im_size': [tensor([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,32, 32, 32, 32, 32, 32, 32, 32,
#                                   , 'index': tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
#                                                        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
#                                                        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
#                                                        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
#                                                        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
#                                                        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
#                                                        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
#                                                        98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
#                                                        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
#                                                        126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
#                                                        140, 141, 142, 143, 144, 145, 146, 147, 一直到511，那应该就是batch 512个
# 一共就是‘image’，‘target’，‘meta’{},''