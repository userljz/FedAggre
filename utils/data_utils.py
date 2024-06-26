import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10, CIFAR100
import logging
from logging import debug, info

logging.basicConfig(format='%(levelname)s | %(funcName)s | %(lineno)d: %(message)s', level=logging.INFO)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class dsDict:
    def __init__(self, ds, mean, std):
        self.dataset = ds
        self.mean = mean
        self.std = std
        # self.train_transform = transforms.Compose([
        #     transforms.Pad(4, padding_mode='reflect'),
        #     transforms.RandomCrop(32),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=self.mean, std=self.std)
        # ])
        self.train_transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        self.test_transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])


cifar10_dict = dsDict(CIFAR10, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
cifar100_dict = dsDict(CIFAR100, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def dirichlet_split_noniid(args, train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max() + 1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1] * len(k_idcs)).
                                                  astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def load_dataloader(args, dataset_name, dataroot, is_iid=1, dataloader_num=1):
    '''
    Example:

    '''
    if dataset_name == 'cifar10':
        ds = cifar10_dict
    elif dataset_name == 'cifar100':
        ds = cifar100_dict

    if is_iid == 1 and dataloader_num == 1:
        train_loader = torch.utils.data.DataLoader(
            ds.dataset(root=dataroot, transform=ds.train_transform, train=True, download=True),
            batch_size=args.cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            ds.dataset(root=dataroot, transform=ds.test_transform, train=False, download=True),
            batch_size=args.cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, test_loader

    elif dataloader_num > 1:
        trainset = ds.dataset(root=dataroot, transform=ds.train_transform, train=True, download=True)
        testset = ds.dataset(root=dataroot, transform=ds.test_transform, train=False, download=True)
        if is_iid == 1:
            # return normal iid multi-clients trainloaders
            partition_size = len(trainset) // args.cfg.num_clients
            lengths = [partition_size] * args.cfg.num_clients
            datasets = random_split(trainset, lengths, torch.Generator().manual_seed(args.cfg.seed))

            trainloaders = []
            valloaders = []
            for i in datasets:
                len_val = len(i) // 10  # 10 % validation set
                len_train = len(i) - len_val
                lengths = [len_train, len_val]
                ds_train, ds_val = random_split(i, lengths, torch.Generator().manual_seed(args.cfg.seed))
                trainloaders.append(DataLoader(ds_train, batch_size=args.cfg.batch_size, shuffle=True))
                valloaders.append(DataLoader(ds_val, batch_size=args.cfg.batch_size))
            testloader = DataLoader(testset, batch_size=args.cfg.batch_size)
            return trainloaders, valloaders, testloader

        elif is_iid == 0:
            # return non-iid multi-clients trainloaders
            labels = np.array(trainset.targets)
            client_idcs = dirichlet_split_noniid(args, labels, args.cfg.dirichlet_alpha, args.cfg.num_clients)
            client_trainsets = []
            for client_i in client_idcs:
                client_trainsets.append(Subset(trainset, client_i))

            train_loaders = []
            val_loaders = []
            for i in client_trainsets:
                len_val = len(i) // 10  # 10 % validation set
                len_train = len(i) - len_val
                lengths = [len_train, len_val]
                ds_train, ds_val = random_split(i, lengths, torch.Generator().manual_seed(args.cfg.seed))
                train_loaders.append(DataLoader(ds_train, batch_size=args.cfg.batch_size, shuffle=True))
                val_loaders.append(DataLoader(ds_val, batch_size=args.cfg.batch_size))

            # trainloaders = [DataLoader(d, batch_size=args.cfg.batch_size, shuffle=True) for d in client_trainsets]
            # testloaders = [DataLoader(d, batch_size=args.cfg.batch_size, shuffle=True) for d in client_testsets]
            return train_loaders, val_loaders
