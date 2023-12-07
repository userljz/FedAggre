from Utils.typing_utils import FitRes, FitIns
import os
import sys
from collections import OrderedDict
from typing import List, Optional, Tuple
from addict import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, Subset
from collections import defaultdict


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
        image_features = features.float().to(self.args.device)

        if self.args.cfg.use_softmax == 1:
            logits_per_image = (100 * image_features @ self.text_emb.t()).softmax(dim=-1)
        else:
            logits_per_image = 100 * image_features @ self.text_emb.t()  # [bsz, num_class]

        logits_per_image = logits_per_image / logits_per_image.norm(dim=-1, keepdim=True)
        logits_per_image = torch.where(target == 1, 1 - logits_per_image, logits_per_image - self.args.cfg.margin)
        logits_per_image = torch.clamp(logits_per_image, min=0)
        loss = torch.sum(logits_per_image)  # [1,]

        return loss



def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net

def train(args, train_loader, model, criterion, optimizer, epoch, cid, config, extra_loss_embedding, criterion_extra,
          before_fc_emb=None, after_fc_emb=None):
    """one epoch training"""
    losses = AverageMeter()
    for idx, (images, labels) in enumerate(train_loader):
        # ------ Use Before Fc Emb ------
        if args.cfg.use_before_fc_emb:
            before_fc_emb = args.catemb.update(before_fc_emb, images, labels)

        # ------ Calculate Feats
        images, labels = images.to(args.device), labels.to(args.device)
        features = model.get_img_feat(images)

        # ------ Use After Fc Emb ------
        if args.cfg.use_after_fc_emb:
            after_fc_emb = args.catemb.update(after_fc_emb, features, labels)

        # ------ Compute traditional_loss ------
        traditional_loss_num = 0
        traditional_loss = 0
        if config['server_round'] > args.cfg.use_traditional_loss_lower_limit:
            if args.cfg.use_traditional_loss_upper_limit == 0 or config[
                'server_round'] < args.cfg.use_traditional_loss_upper_limit:
                traditional_loss = criterion(features, labels)
                traditional_loss_num = labels.shape[0]

        # ------ Compute loss from average features ------
        extra_loss_num = 0
        extra_loss = 0
        extra_loss_weight = 0
        if args.cfg.use_after_fc_emb == 1 and config[
            'server_round'] > args.cfg.use_extra_emb_round:  # make sure it's not the first round
            extra_loss = criterion_extra(features, labels)
            extra_loss_num = labels.shape[0]
            extra_loss_weight = args.cfg.extra_loss_weight

        # ------ Loss Summary ------
        loss = extra_loss_weight * extra_loss + (1 - extra_loss_weight) * traditional_loss

        # update metric
        bsz = int((1 - extra_loss_weight) * traditional_loss_num + extra_loss_weight * extra_loss_num)
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, before_fc_emb, after_fc_emb


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
    def update(self, embdict, img_emb, label):
        """
        Update the img_emb and label into embdict
        Lables serve as keys and the Embeddings serve as values

        """
        img_emb, label = img_emb.detach().cpu(), label.detach().cpu()

        for index, label_i in enumerate(label):
            embdict[label_i.item()].append(img_emb[index])

        return embdict

    def avg(self, dictin):
        """
        In a dictionary, a key usually has multiple values. This function is used to average the multiple values.
        (optional with or without weighting)
        Return:
                {'label1': ([averaged emb], emb num before averaging), ...}
        """
        dict_return = defaultdict(list, {})
        for k, v in dictin.items():
            v = torch.stack(tuple(v))
            _num = v.size()[0]
            v = torch.mean(v, dim=0)
            dict_return[k] = (v, _num)
        return dict_return

    def generate_train_emb(self, CatEmbDict_avg):
        """
        First sort the average embedding, in the sort of classes

        Return: [n,1024], [n]
        """
        CatEmbDict_avg = {k: v[0] for k, v in CatEmbDict_avg.items()}
        _img_emb = torch.stack(tuple(CatEmbDict_avg.values()), dim=0)
        _labels = torch.Tensor(tuple(CatEmbDict_avg.keys())).long()
        sorting_indices = torch.argsort(_labels)

        # 使用这些索引对第一个tensor进行排序
        sorted_tensor = _img_emb[sorting_indices]
        return sorted_tensor

    def merge(self, dict_origin, dict_new):
        """
        Merge dict_new into dict_origin
        Both dict_origin and dict_new has the format of {'label1': ([averaged emb], emb num before averaging), ...}
        """
        dict_origin_keys = dict_origin.keys()
        for k1, v1 in dict_new.items():
            if k1 in dict_origin_keys:
                _new_value1 = dict_origin[k1][1] + v1[1]
                _new_value0 = (dict_origin[k1][0] * dict_origin[k1][1] + v1[0] * v1[1]) / _new_value1
                dict_origin[k1] = (_new_value0, _new_value1)
            else:
                dict_origin[k1] = v1
        return dict_origin