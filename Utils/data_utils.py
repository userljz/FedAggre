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
from addict import Dict
import random


# logging.basicConfig(format='%(levelname)s | %(funcName)s | %(lineno)d: %(message)s', level=logging.INFO)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class dsDict:
    def __init__(self, ds, mean, std):
        self.dataset = ds
        self.mean = mean
        self.std = std

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
    
    # 后处理:防止某个Client一个样本也没有分到, 保证每个Client至少分到一个Sample
    sorted_client_idcs = sorted(client_idcs, key=len)
    
    client_idcs_without_0 = []
    used_data_from_other_client = 0
    for idx, client_i in enumerate(sorted_client_idcs[:-1]):
        if len(client_i) == 0:
            _client_i = np.array([sorted_client_idcs[-1][used_data_from_other_client]])
            used_data_from_other_client += 1
            client_idcs_without_0.append(_client_i)
        else:
            client_idcs_without_0.append(client_i)
    client_idcs_without_0.append(sorted_client_idcs[-1][used_data_from_other_client:])    
    return client_idcs_without_0


def load_dataloader(args, dataset_name, dataroot, is_iid=1, dataloader_num=1):
    '''
    Example:

    '''
# ============================================================================ #
# Select Dataset
    if dataset_name == 'cifar10':
        ds = cifar10_dict
    elif dataset_name == 'cifar100':
        ds = cifar100_dict

# ============================================================================ #
# Centralized DataLoader
    if is_iid == 1 and dataloader_num == 1:
        train_loader = torch.utils.data.DataLoader(
            ds.dataset(root=dataroot, transform=ds.train_transform, train=True, download=True),
            batch_size=args.cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            ds.dataset(root=dataroot, transform=ds.test_transform, train=False, download=True),
            batch_size=args.cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, test_loader

# ============================================================================ #
# FL 
    elif dataloader_num > 1:
        trainset = ds.dataset(root=dataroot, transform=ds.train_transform, train=True, download=True)
        testset = ds.dataset(root=dataroot, transform=ds.test_transform, train=False, download=True)
        testloader = DataLoader(testset, batch_size=args.cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # if is_iid == 1:
        #     # return normal iid multi-clients trainloaders
        #     partition_size = len(trainset) // args.cfg.num_clients
        #     lengths = [partition_size] * args.cfg.num_clients
        #     datasets = random_split(trainset, lengths, torch.Generator().manual_seed(args.cfg.seed))

        #     trainloaders = []
        #     valloaders = []
        #     for i in datasets:
        #         len_val = len(i) // 10  # 10 % validation set
        #         len_train = len(i) - len_val
        #         lengths = [len_train, len_val]
        #         ds_train, ds_val = random_split(i, lengths, torch.Generator().manual_seed(args.cfg.seed))
        #         trainloaders.append(DataLoader(ds_train, batch_size=args.cfg.batch_size, shuffle=True))
        #         valloaders.append(DataLoader(ds_val, batch_size=args.cfg.batch_size))
        #     return trainloaders, valloaders, testloader

        # elif is_iid == 0:
            # return non-iid multi-clients trainloaders

        labels = np.array(trainset.targets)
        client_idcs = dirichlet_split_noniid(args, labels, args.cfg.dirichlet_alpha, args.cfg.num_clients)
        client_trainsets = []
        for client_i in client_idcs:
            client_trainsets.append(Subset(trainset, client_i))

        train_loaders = []
        # val_loaders = []
        for i in client_trainsets:
            # len_val = len(i) // 10  # 10 % validation set
            # len_train = len(i) - len_val
            # lengths = [len_train, len_val]
            # ds_train, ds_val = random_split(i, lengths, torch.Generator().manual_seed(args.cfg.seed))
            train_loaders.append(DataLoader(i, batch_size=args.cfg.batch_size, shuffle=True))
            # val_loaders.append(DataLoader(ds_val, batch_size=args.cfg.batch_size))

        # trainloaders = [DataLoader(d, batch_size=args.cfg.batch_size, shuffle=True) for d in client_trainsets]
        # testloaders = [DataLoader(d, batch_size=args.cfg.batch_size, shuffle=True) for d in client_testsets]
        return train_loaders, testloader


# 20240103 Currently Use
def load_dataloader_from_generate(args, dataset_name, dataloader_num=1):
    set_random_seed()
    if dataset_name == 'cifar10':
        test_data_per_class = 1000
        train_img = torch.load('/home/ljz/dataset/cifar10_generated/cifar10Train_RN50_imgembV1.pth')
        train_label = torch.load('/home/ljz/dataset/cifar10_generated/cifar10Train_labelsV1.pth')
        train_img = train_img.float()
        train_img_label_list = [(train_img[i], train_label[i]) for i in range(len(train_label))]

        test_img = torch.load('/home/ljz/dataset/cifar10_generated/cifar10Test_RN50_imgembV1.pth')
        test_label = torch.load('/home/ljz/dataset/cifar10_generated/cifar10Test_labelsV1.pth')
        test_img = test_img.float()
        test_img_label_list = [(test_img[i], test_label[i]) for i in range(len(test_label))]

    elif dataset_name == 'PathMNIST' or dataset_name == 'OrganAMNIST' or dataset_name == 'emnist62':
        if args.cfg.only_test_training_labels == 1:
            print('Please specify the test_data_per_class')
        if args.cfg.model_name == 'ViT-B/32':
            train_img = torch.load(f'/home/ljz/dataset/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Train_imgemb.pth')
            train_label = torch.load(f'/home/ljz/dataset/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Train_labels.pth')
            train_img = train_img.float()
            train_img_label_list = [(train_img[i], train_label[i]) for i in range(len(train_label))]

            test_img = torch.load(f'/home/ljz/dataset/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Test_imgemb.pth')
            test_label = torch.load(f'/home/ljz/dataset/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Test_labels.pth')
            test_img = test_img.float()
            test_img_label_list = [(test_img[i], test_label[i]) for i in range(len(test_label))]


    elif dataset_name == 'emnist':
        test_data_per_class = 400
        if args.cfg.model_name == 'ViT-B/32':
            train_img = torch.load('/home/ljz/dataset/emnist_generated_vitb32/emnist_vitb32Train_imgemb.pth')
            train_label = torch.load('/home/ljz/dataset/emnist_generated_vitb32/emnist_vitb32Train_labels.pth')
            train_img = train_img.float()
            train_img_label_list = [(train_img[i], train_label[i]) for i in range(len(train_label))]

            test_img = torch.load('/home/ljz/dataset/emnist_generated_vitb32/emnist_vitb32Test_imgemb.pth')
            test_label = torch.load('/home/ljz/dataset/emnist_generated_vitb32/emnist_vitb32Test_labels.pth')
            test_img = test_img.float()
            test_img_label_list = [(test_img[i], test_label[i]) for i in range(len(test_label))]


    elif dataset_name == 'cifar100':
        test_data_per_class = 100
# ============================================================================ #
# CLIP RN50 Embedding
        if args.cfg.model_name == 'RN50':
            train_img = torch.load('/home/ljz/dataset/cifar100_generated/cifar100Train_RN50_imgembV1.pth')
            train_label = torch.load('/home/ljz/dataset/cifar100_generated/cifar100Train_labelsV1.pth')
            train_img = train_img.float()
            train_img_label_list = [(train_img[i], train_label[i]) for i in range(len(train_label))]

            test_img = torch.load('/home/ljz/dataset/cifar100_generated/cifar100Test_RN50_imgembV1.pth')
            test_label = torch.load('/home/ljz/dataset/cifar100_generated/cifar100Test_labelsV1.pth')
            test_img = test_img.float()
            test_img_label_list = [(test_img[i], test_label[i]) for i in range(len(test_label))]

# ============================================================================ #
# CLIP ViT-B/32 Embedding
        elif args.cfg.model_name == 'ViT-B/32':
            train_img = torch.load('/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32Train_imgemb.pth')
            train_label = torch.load('/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32Train_labels.pth')
            train_img = train_img.float()
            train_img_label_list = [(train_img[i], train_label[i]) for i in range(len(train_label))]

            test_img = torch.load('/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32Test_imgemb.pth')
            test_label = torch.load('/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32Test_labels.pth')
            test_img = test_img.float()
            test_img_label_list = [(test_img[i], test_label[i]) for i in range(len(test_label))]

# ============================================================================ #
# Timm Pre-trained RN50 Embedding
        # train_img = torch.load('/home/ljz/dataset/cifar100_gene_TimmRN50/TrainImg_PretrainedRN50_CIFAR100.pth')
        # train_label = torch.load('/home/ljz/dataset/cifar100_gene_TimmRN50/TrainLabel_PretrainedRN50_CIFAR100.pth')
        # train_img = train_img.float()
        # train_img_label_list = [(train_img[i], train_label[i]) for i in range(len(train_label))]

        # test_img = torch.load('/home/ljz/dataset/cifar100_gene_TimmRN50/TestImg_PretrainedRN50_CIFAR100.pth')
        # test_label = torch.load('/home/ljz/dataset/cifar100_gene_TimmRN50/TestLabel_PretrainedRN50_CIFAR100.pth')
        # test_img = test_img.float()
        # test_img_label_list = [(test_img[i], test_label[i]) for i in range(len(test_label))]
    else:
        print('Please specify the dataset')

    
    if dataloader_num == 1:
        train_loader = torch.utils.data.DataLoader(
            train_img_label_list,
            batch_size=args.cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            test_img_label_list,
            batch_size=args.cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return [train_loader], test_loader

    elif dataloader_num > 1:
        test_loader = torch.utils.data.DataLoader(
            test_img_label_list,
            batch_size=args.cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # return non-iid multi-clients trainloaders
        labels = np.array(train_label)
        client_idcs = dirichlet_split_noniid(args, labels, args.cfg.dirichlet_alpha, dataloader_num)
        client_trainsets = []
        for client_i in client_idcs:
            client_trainsets.append(Subset(train_img_label_list, client_i))
        
        # print(f'{client_trainsets[0][0]}')
        # for trainset_i in client_trainsets:
        #     label = [sample[1].item() for sample in trainset_i]
        #     label = set(label)
        #     print(label)
        #     print(len(label))
        

        train_loaders = []
        val_loaders = []
        for i in client_trainsets:
            train_loaders.append(DataLoader(i, batch_size=args.cfg.batch_size, shuffle=True))

        if args.cfg.only_test_training_labels == 1:
            
            test_img_label_list_sorted = sorted(test_img_label_list, key=lambda x: x[1].item())
            
            train_label_list = []
            for trainset_i in client_trainsets:
                label = [sample[1].item() for sample in trainset_i]
                label = list(set(label))
                train_label_list.append(label)
            
            client_testsets = []
            for client_i in train_label_list:
                label_index = []
                for label_i in client_i:
                    lower_limit = label_i * test_data_per_class
                    upper_limit = (label_i+1) * test_data_per_class
                    label_index.extend([i for i in range(lower_limit , upper_limit)])
                client_testsets.append(Subset(test_img_label_list_sorted, label_index))

            test_loader = []
            for i in client_testsets:
                test_loader.append(DataLoader(i, batch_size=args.cfg.batch_size, shuffle=False))

        return train_loaders, test_loader


def label_collect(test_loader):
    label_list = []
    for img, label in test_loader:
        label_list.append(label)
    label_list = torch.cat(label_list)
    label_list = [i.item() for i in label_list]
    label_list = set(label_list)
    return label_list

if __name__ == '__main__':
    args = Dict()
    args.cfg.dirichlet_alpha = 0.1
    args.cfg.num_clients = 100
    args.cfg.batch_size = 32
    dataset_name = 'cifar100'
    args.cfg.model_name = 'ViT-B/32'
    train_loader_list, test_loader = load_dataloader_from_generate(args, dataset_name, dataloader_num=10)
    for index,train_loader_i in enumerate(train_loader_list):
        label = label_collect(train_loader_i)
        print(f'[{index}]({len(label)}){label=}')

    #     # print(f'{len(train_loader_i)=}')
    #     for i in train_loader_i:
    #         # print(len(i))

    # print(f'{len(test_loader)=}')