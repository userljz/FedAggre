import os
import sys
from collections import OrderedDict
from typing import List, Optional, Tuple
from addict import Dict
from utils.log_utils import cus_logger

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, Subset
from collections import defaultdict

import flwr as fl
import wandb

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning:.
It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, args, text_emb):
        super(SupConLoss, self).__init__()
        self.args = args
        self.text_emb = text_emb  # text_emb: embedding of different classes [num_class, image_feats]

    def forward(self, features, labels=None):
        """
        Args:
            features: hidden vector of shape [bsz, image_feats].
            labels: ground truth of shape [bsz].

        Returns:
            A loss scalar.
        """
        # --- generate one-hot target ---
        num_classes = self.args.cfg.num_class
        target = F.one_hot(labels, num_classes).to(self.args.device)

        # image_features = features / features.norm(dim=1, keepdim=True)
        image_features = features.float()

        if self.args.cfg.use_softmax == 1:
            logits_per_image = (100 * image_features @ self.text_emb.t()).softmax(dim=-1)
        else:
            logits_per_image = 100 * image_features @ self.text_emb.t()  # [bsz, num_class]

        logits_per_image = logits_per_image / logits_per_image.norm(dim=-1, keepdim=True)
        logits_per_image = torch.where(target == 1, 1 - logits_per_image, logits_per_image - self.args.cfg.margin)
        logits_per_image = torch.clamp(logits_per_image, min=0)
        loss = torch.sum(logits_per_image)  # [1,]

        return loss


def train(args, train_loader, model, criterion, optimizer, epoch, cid, config, extra_loss_embedding, criterion_extra):
    """one epoch training"""
    losses = AverageMeter()
    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        features = model.get_img_feat(images)
        args.catemb.update(features, labels)  # collect features
        
        # ------ Compute traditional_loss ------
        traditional_loss_num = 0
        traditional_loss = 0
        if config['server_round'] > args.cfg.use_traditional_loss_lower_limit:
            if args.cfg.use_traditional_loss_upper_limit == 0 or config['server_round'] < args.cfg.use_traditional_loss_upper_limit:
                traditional_loss = criterion(features, labels)
                traditional_loss_num = labels.shape[0]
            
        
        # ------ Compute loss from average features ------
        extra_loss_num = 0
        extra_loss = 0
        extra_loss_weight = 0
        if args.cfg.use_extra_emb == 1 and config['server_round'] > args.cfg.use_extra_emb_round:  # make sure it's not the first round
            extra_loss = criterion_extra(features, labels)
            extra_loss_num = labels.shape[0]
            extra_loss_weight = args.cfg.extra_loss_weight

        loss = extra_loss_weight * extra_loss + (1 - extra_loss_weight) * traditional_loss

        # update metric
        bsz = int((1 - extra_loss_weight) * traditional_loss_num + extra_loss_weight * extra_loss_num)
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def test(args, net, testloader, criterion):
    """Evaluate the network on the entire test set."""
    losses = AverageMeter()
    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(args.device), labels.to(args.device)
            features = net.get_img_feat(images)

            loss = criterion(features, labels)
            losses.update(loss, labels.shape[0])

            # similarity = F.cosine_similarity(features.unsqueeze(1), args.text_emb.unsqueeze(0), dim=2)
            similarity = features @ (args.text_emb.t())
            _, predicted_indices = torch.max(similarity, dim=1)
            total += labels.shape[0]
            correct += (predicted_indices == labels).sum().item()

    _accuracy = correct / total
    return losses.avg, _accuracy


def get_parameters(net) -> List[np.ndarray]:
    print('*** in get_param ***')
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, args=None):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.args = args
        self.client_logger = cus_logger(args, 'client_logger')
        label_counts = self.count_labels(trainloader)
        self.client_logger.info(f'Cid_{self.cid}, label_counts:{label_counts}')
        self.criterion = SupConLoss(self.args, self.args.text_emb)
        wandb.init(project=args.cfg["wandb_project"], name=args.cfg["logfile_info"], config=args.cfg)

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)

        optimizer = torch.optim.SGD(self.net.parameters(),
                                    lr=float(self.args.cfg.local_lr),
                                    momentum=float(self.args.cfg.local_momentum),
                                    weight_decay=float(self.args.cfg.local_weight_decay))

        # ------ train multiple epochs ------
        if config['server_round'] > self.args.cfg.use_extra_emb_round:
            extra_loss_embedding = self.args.catemb.generate_train_emb(config['prototype_avg'])
            extra_loss_embedding = extra_loss_embedding.to(self.args.device)
            criterion_extra = SupConLoss(self.args, extra_loss_embedding)
        else:
            extra_loss_embedding, criterion_extra = None, None

        for epoch in range(1, self.args.cfg.local_epoch + 1):
            loss = train(self.args, self.trainloader, self.net, self.criterion, optimizer, epoch, self.cid, config, extra_loss_embedding, criterion_extra)
            self.client_logger.info(
                f"Train_Client{self.cid}:[{epoch}/{self.args.cfg.local_epoch}], Train_loss:{loss}, DataLength:{len(self.trainloader.dataset)}")
            wandb.log({f"Train_Client{self.cid}|Train_loss:": loss})

        # ------ test accuracy after training ------
        loss, accuracy = test(self.args, self.net, self.valloader, self.criterion)
        self.client_logger.info(f'Val_Client_{self.cid}, Accu:{accuracy}')
        wandb.log({f"Val_Client{self.cid}|Accu:": accuracy})

        # ------ save client model ------
        if self.args.cfg.save_client_model == 1:
            torch.save(self.net, f"./save_client_model/client_{self.cid}.pth")
            self.client_logger.info(f'Finish saving client model {self.cid}')

        # ------ Use Extra Embedding ------
        if self.args.cfg.use_extra_emb == 1:
            CatEmbDict_avg, emb_num = self.args.catemb.avg()
            # CatEmbDict_avg = self.args.catemb.CatEmbDict
        else:
            CatEmbDict_avg, emb_num = 0, 0

        return get_parameters(self.net), len(self.trainloader), {'prototype_avg': CatEmbDict_avg, 'prototype_num': emb_num}

    def evaluate(self, parameters, config):
        # self.client_logger.info('********** enter evaluate of FlowerClient **********')
        set_parameters(self.net, parameters)
        loss, _accuracy = test(self.args, self.net, self.valloader, self.criterion)
        self.client_logger.info(f'Cid:{self.cid}, Accu:{float(_accuracy)}, Test_length:{len(self.valloader)}')  # ljz
        return float(loss), len(self.valloader), {"accuracy": float(_accuracy), 'test': 12138}  # ljz

    def count_labels(self, data_loader):
        # Initialize a counter dictionary
        label_counts = {}

        for _, labels in data_loader:
            # If labels are not already on CPU, move them
            labels = labels.cpu()
            for label in labels:
                # If label is tensor, convert to python number
                if isinstance(label, torch.Tensor):
                    label = label.item()
                # Increment the count for this label
                label_counts[label] = label_counts.get(label, 0) + 1

        return label_counts


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CategoryEmbedding(object):
    def __init__(self):
        self.reset()
        

    def reset(self):
        self.CatEmbDict = defaultdict(list, {})
        self.CatEmbDict_avg = defaultdict(list, {})
        self.CatEmbDict_merge = defaultdict(list, {})
        self.emb_num = defaultdict(list, {})
        self.num_sum = defaultdict(int)

    def update(self, img_emb, label):
        img_emb = img_emb.detach()
        for index, label_i in enumerate(label):
            self.CatEmbDict[label_i.item()].append(img_emb[index])

    def avg(self, dictin=None, dict_num=None):
        if dictin is None:
            for k,v in self.CatEmbDict.items():
                v = torch.stack(tuple(v))
                self.emb_num[k] = v.size()[0]
                v = torch.mean(v, dim=0)
                self.CatEmbDict_avg[k] = v
            return self.CatEmbDict_avg, self.emb_num
        else:
            dictout = defaultdict(list, {})
            for k,v in dictin.items():
                v = torch.stack(tuple(v))
                v = torch.sum(v, dim=0)
                v = v / dict_num[k]
                dictout[k] = v
            # print(f'{dict_num =}')
            # print(f'{dictout =}')
            return dictout

    def generate_train_emb(self, CatEmbDict_avg):
        """
        First sort the average embedding, in the sort of classes

        Return: [n,1024], [n]
        """
        _img_emb = torch.stack(tuple(CatEmbDict_avg.values()), dim=0)
        _labels = torch.Tensor(tuple(CatEmbDict_avg.keys())).long()
        sorting_indices = torch.argsort(_labels)

        # 使用这些索引对第一个tensor进行排序
        sorted_tensor = _img_emb[sorting_indices]
        return sorted_tensor

    def merge(self, dict_avg, dict_num):
        for k1, v1 in dict_avg.items():
            _num = dict_num[k1]
            self.num_sum[k1] += _num
            self.CatEmbDict_merge[k1].append(v1 * _num)

        

