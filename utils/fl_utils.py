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

        image_features = features / features.norm(dim=1, keepdim=True)
        image_features = image_features.float()
        logits_per_image = 100 * image_features @ self.text_emb.t()  # [bsz, num_class]
        logits_per_image = logits_per_image / logits_per_image.norm(dim=-1, keepdim=True)
        logits_per_image = torch.where(target == 1, 1 - logits_per_image, logits_per_image - self.args.cfg.margin)
        logits_per_image = torch.clamp(logits_per_image, min=0)
        loss = torch.sum(logits_per_image)  # [1,]

        return loss


def train(args, train_loader, model, criterion, optimizer, epoch, cid):
    """one epoch training"""

    losses = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(args.device), labels.to(args.device)

        # compute loss
        features = model.get_img_feat(images)
        loss = criterion(features, labels)

        # update metric
        bsz = labels.shape[0]
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        # if (idx + 1) % 100 == 0:
        #     custom_logger.info(f'Cid{cid}Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t loss {losses.val:.3f} ({losses.avg:.3f})')
        #     sys.stdout.flush()

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


def get_parameters(net):
    ret = []
    for name, val in net.state_dict().items():
        if 'fc_cus' in name:
            ret.append(val.cpu().numpy())
    return ret


def set_parameters(net, parameters):
    # params_dict = zip(net.state_dict().keys(), parameters)
    # # state_dict = OrderedDict     ({k: torch.Tensor(v) for k, v in params_dict})
    # state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    # net.load_state_dict(state_dict, strict=True)
    # return net
    """
    :param net: CLIP(with pretrained params) + fc_cus(random initialized)
    :param parameters: Only include the last layer fc_cus, dict
    :return: CLIP with new params loaded in
    """
    fc_key = []
    for k in net.state_dict().keys():
        if 'fc_cus' in k:
            fc_key.append(k)
    parameters = [torch.from_numpy(p) for p in parameters]

    params_dict = dict(zip(fc_key, parameters))
    # print(f'{params_dict = }')
    # for k,v in params_dict.items():
    #     print(f'{v.dtype=}')
    pretrain_dict = net.state_dict()
    pretrain_dict.update(params_dict)
    net.load_state_dict(pretrain_dict)
    return net


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
        # print(f'Flower Client Params: {parameters}')
        model = set_parameters(self.net, parameters)

        # self.criterion = SupConLoss(self.args, self.args.text_emb)
        model.cus_train()
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(pg,
                                    lr=float(self.args.cfg.local_lr),
                                    momentum=float(self.args.cfg.local_momentum),
                                    weight_decay=float(self.args.cfg.local_weight_decay))

        # ------ train multiple epochs ------
        for epoch in range(1, self.args.cfg.local_epoch + 1):
            loss = train(self.args, self.trainloader, self.net, self.criterion, optimizer, epoch, self.cid)
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

        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.client_logger.info('********** enter evaluate of FlowerClient **********')
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


