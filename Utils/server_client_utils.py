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

def train(args, train_loader, model, criterion, optimizer, epoch, cid, config, p_images_group=None, p_labels_group=None):
    """one epoch training"""
    losses = AverageMeter()
    for idx, (images, labels) in enumerate(train_loader):
        # ------ Calculate Feats ------
        images, labels = images.to(args.device), labels.to(args.device)
        features = model.get_img_feat(images)

        # ------ Compute traditional_loss ------
        traditional_loss = criterion(features, labels)
        traditional_loss_num = labels.shape[0]
        loss = traditional_loss
        bsz = traditional_loss_num

        # ------ Compute loss from Before Fc Emb ------
        
        if args.cfg.use_before_fc_emb == 1 and config['server_round'] > 1:  # make sure it's not the first round
            
            if idx < len(p_labels_group):
                p_images, p_labels = p_images_group[idx], p_labels_group[idx]
                p_images, p_labels = p_images.to(args.device), p_labels.to(args.device)
                p_features = model.get_img_feat(p_images)
                p_loss = criterion(p_features, p_labels)
                p_loss_num = p_labels.shape[0]
                
                loss = loss + p_loss
                bsz = bsz + p_loss_num

        # update metric
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



def train_baseline(args, train_loader, model, criterion, optimizer, epoch, cid, config, p_images_group=None, p_labels_group=None, global_model=None):
    """one epoch training"""
    losses = AverageMeter()
    
    
    for idx, (images, labels) in enumerate(train_loader):
        
        # ------ Calculate Feats ------
        images, labels = images.to(args.device), labels.to(args.device)
        features = model(images)  # [bz, 100]
        
        if args.cfg.strategy == 'FedProx' and config['server_round'] > 1:
            proximal_term = 0.0
            proximal_mu = args.cfg.proximal_mu
            for local_weights, global_weights in zip(model.parameters(), global_model.parameters()):
                proximal_term += torch.square((local_weights - global_weights).norm(2))
            loss = criterion(features, labels) + (proximal_mu / 2) * proximal_term
            bsz = labels.shape[0]
        
        else:
            loss = criterion(features, labels)
            bsz = labels.shape[0]

        

        # ------ Compute loss from Before Fc Emb ------
        
        if args.cfg.use_before_fc_emb == 1 and config['server_round'] > 1:  # make sure it's not the first round
            if idx < len(p_labels_group):
                p_images, p_labels = p_images_group[idx], p_labels_group[idx]
                p_images, p_labels = p_images.to(args.device), p_labels.to(args.device)
                p_features = model.get_img_feat(p_images)
                p_loss = criterion(p_features, p_labels)
                p_loss_num = p_labels.shape[0]
                
                loss = loss + p_loss
                bsz = bsz + p_loss_num

        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def test_baseline(args, net, testloader, criterion):
    """Evaluate the network on the entire test set."""
    losses = AverageMeter()
    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(args.device), labels.to(args.device)
            features = net(images)

            loss = criterion(features, labels)
            losses.update(loss, labels.shape[0])

            # similarity = F.cosine_similarity(features.unsqueeze(1), args.text_emb.unsqueeze(0), dim=2)
            _, predicted_indices = torch.max(features, dim=1)
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
        Both dict_origin and dict_new has the format of {'label1': ([averaged emb], emb_num_before_averaging), ...}
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

    def merge_mean_var(sefl, dict_origin, dict_new):
        """
        {'label1': ([emb_avg, emb_var], emb_num)
        """
        dict_origin_keys = dict_origin.keys()
        for k1, v1 in dict_new.items():
            if k1 in dict_origin_keys:
                n1 = dict_origin[k1][1]
                n2 = v1[1]
                mean1 = dict_origin[k1][0][0]
                mean2 = v1[0][0]
                var1 = dict_origin[k1][0][1]
                var2 = v1[0][1]

                combined_mean = (n1 * mean1 + n2 * mean2) / (n1 + n2)
                combined_var = ((n1 * var1 + n2 * var2) + ((n1 * n2) / (n1 + n2)) * (mean1 - mean2) ** 2) / (n1 + n2)
                
                dict_origin[k1] = ([combined_mean, combined_var], n1+n2)
            else:
                dict_origin[k1] = v1
        return dict_origin

    def avg_mean_var(self, dictin):
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
            v_mean = v.mean(dim=0)
            v_var = v.var(dim=0, unbiased=False)
            dict_return[k] = ([v_mean, v_var], _num)
        return dict_return

    def generate_pseudo_data(self, CatEmbDict_avg, gene_num, batch_size, more_dense=1):
        """
        First sort the average embedding, in the sort of classes

        CatEmbDict_avg: {'label1': ([mean, var], num), ...}

        Return: [gene_num, 1024], [gene_num]
        """
        p_sample_list = []
        p_label_list = []
        for k, v in CatEmbDict_avg.items():
            mean = v[0][0]
            var = v[0][1]
            std_dev = torch.sqrt(var) / more_dense
            pseudo_samples = [torch.normal(mean, std_dev) for _ in range(int(gene_num/9))]  # [num, 1024]
            pseudo_samples = torch.stack(pseudo_samples, dim=0)  # [num, 1024]
            pseudo_labels = torch.full((int(gene_num/9),), k)  # [num,]

            p_sample_list.append(pseudo_samples)  # [[num,1024], [num, 1024], ...]
            p_label_list.append(pseudo_labels)  # [[num,], [num, ], ...]
       
        p_sample_cat = torch.cat(p_sample_list, dim=0)
        p_label_cat = torch.cat(p_label_list, dim=0)

        perm = torch.randperm(len(p_label_cat))
        perm = perm[:gene_num]

        # print(f'{perm=}')
        # print(f'{len(perm)=}')
        p_sample_list = [p_sample_cat[i] for i in perm]
        p_label_list = [p_label_cat[i] for i in perm]

        # print(f'{len(p_label_list)=}')
        # print(f'{len(p_sample_list)=}')
        # print(f'{p_sample_list[0].size()=}')

        p_sample_group = [torch.stack(p_sample_list[i:i+batch_size]) for i in range(0, len(p_sample_list), batch_size)]
        p_label_group = [torch.stack(p_label_list[i:i+batch_size]) for i in range(0, len(p_label_list), batch_size)]

        # p_sample_list = torch.cat(p_sample_list, dim=0)
        # p_label_list = torch.cat(p_label_list, dim=0)

        # print(f'{len(p_sample_group)=}')
        # print(f'{p_sample_group[0].size()=}')
        return p_sample_group, p_label_group

def test_vae(net, testloader, device='cuda', shuffle=1, noise=1):
    """Evaluate the network on the entire test set."""
    mean_sum, dist_sum, total_num = 0, 0, 0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            
            images = images.to(device)
            images_s = images
            features = net.generate(images_s, shuffle=shuffle, noise=noise)
            similarity = F.cosine_similarity(features, images, dim=1)
            l2_distances = F.pairwise_distance(features, images, p=2)
            
            mean = torch.sum(similarity)
            dist = torch.sum(l2_distances)
            mean_sum += mean
            dist_sum += dist
            total_num += len(labels)
        return mean_sum/total_num, dist_sum/total_num


def train_vae(train_loader, vae_model, vae_optimizer, epoch, device):
    """one epoch training"""
    losses_vae = 0
    for idx, (images, labels) in enumerate(train_loader):
        # ------ Calculate Feats ------
        images = images.to(device)
        features = vae_model(images)  # [self.decode(z), input, mu, log_var]
        _loss = vae_model.loss_function(features)
        # SGD
        vae_optimizer.zero_grad()
        _loss['loss'].backward()
        vae_optimizer.step()
        losses_vae += _loss['loss']
    return losses_vae



class EmbVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 ) -> None:
        super(EmbVAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(nn.Linear(in_channels, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim))
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, in_channels))


    def encode(self, input):
        """
        :param input: [num, 512]
        :return: [[num,256], [num,256]]
        """
        result = self.encoder(input)  # 512 -> 256

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent emb into the image space.
        :param z: (Tensor) [num x 512]
        :return: (Tensor) [num x 512]
        """
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.001 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, noise=1):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        if noise > 0:
            v_var = z.var(dim=0, unbiased=False)
            std_dev = torch.sqrt(v_var) / noise
            if std_dev.ne(0).all():
                z = torch.normal(z, std_dev) 
        return [self.decode(z), input, mu, log_var]



    def loss_function(self,
                      args,
                      ) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = 0.00025 # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': kld_loss.detach()}

    # def sample(self,
    #            num_samples,c
    #            current_device):
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = torch.randn(num_samples,
    #                     self.latent_dim)

    #     z = z.to(current_device)

    #     samples = self.decode(z)
    #     return samples

    def generate(self, x, noise=1, shuffle=1):
        """
        return: [num x 512]
        """
        if shuffle == 1:
            x = self.random_shuffle(x)
        return self.forward(x, noise=noise)[0]

    def random_shuffle(self, img_emb):
        """
        Randomly shuffle the corresponding dimensions
        :param img_emb: [num, 512]
        :return: [num, 512]
        """
        shuffled_list = []
        for rol_i in range(img_emb.size(1)):
            shuffled_indices_col1 = torch.randperm(img_emb.size(0))
            shuffled_col1 = img_emb[shuffled_indices_col1, rol_i]
            shuffled_list.append(shuffled_col1)
        shuffled_test_list = torch.stack(shuffled_list, dim=1)
        return shuffled_test_list



class EmbAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 ) -> None:
        super(EmbAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(nn.Linear(in_channels, 256), 
                                     nn.ReLU(),
                                     nn.Linear(256, latent_dim),
                                     ).to('cuda')
        
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, in_channels),
                                     ).to('cuda')



    def encode(self, input):
        """
        :param input: [num, 512]
        :return: [[num,256], [num,256]]
        """
        result = self.encoder(input)  # 512 -> 256
        return result

    def decode(self, z):
        """
        Maps the given latent emb into the image space.
        :param z: (Tensor) [num x 512]
        :return: (Tensor) [num x 512]
        """
        result = self.decoder(z)
        return result


    def forward(self, input, noise=1):
        z = self.encode(input)
        # print('****** Analyze z ******')
        # print(f'{torch.max(z)}')
        # print(f'{torch.min(z)}')
        # print(f'{torch.median(z)}')
        if noise > 0:
            # 假设latent_vector是你的64维潜在向量
            noise = torch.randn(z.size()) * 0.1 # noise_std是噪声的标准差
            noise = noise.to('cuda')
            z = z + noise
            # v_var = z.var(dim=0, unbiased=False)
            # std_dev = torch.sqrt(v_var) / noise
            # if (std_dev != 0).all():
            #     z = torch.normal(z, std_dev) 
        return [self.decode(z), input]



    def loss_function(self,
                      args,
                      label
                      ) -> dict:
        recons = args[0]
        input1 = args[1]

        recons = recons / recons.norm(dim=-1, keepdim=True)
        input1 = input1 / input1.norm(dim=-1, keepdim=True)

        recons_loss =F.mse_loss(recons, input1)

        # target = torch.eye(recons.size(0)).to('cuda')
        similarity_matrix = (label[:, None] == label[None, :]).int().to('cuda')

        logits_per_image = recons @ input1.t()
        loss = logits_per_image / logits_per_image.norm(dim=-1, keepdim=True)
        
        loss_sim = torch.where(similarity_matrix == 1, 1 - loss, loss-0.2)
        loss_sim = torch.clamp(loss_sim, min=0)
        loss_sim = torch.sum(loss_sim)

        loss_sum = loss_sim * 0.1 + recons_loss * 10000

        return {'loss': loss_sum}

    def generate(self, x, noise=1, shuffle=0):
        """
        return: [num x 512]
        """
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            if shuffle == 1:
                x = self.random_shuffle(x)
            x = self.forward(x, noise=noise)[0]
        return x

    def random_shuffle(self, img_emb):
        """
        Randomly shuffle the corresponding dimensions
        :param img_emb: [num, 512]
        :return: [num, 512]
        """
        shuffled_list = []
        for rol_i in range(img_emb.size(1)):
            shuffled_indices_col1 = torch.randperm(img_emb.size(0))
            shuffled_col1 = img_emb[shuffled_indices_col1, rol_i]
            shuffled_list.append(shuffled_col1)
        shuffled_test_list = torch.stack(shuffled_list, dim=1)
        return shuffled_test_list
    

def generate_from_meanvar(CatEmbDict_avg, gene_num, batch_size):
    """
    First sort the average embedding, in the sort of classes

    CatEmbDict_avg: {'label1': ([mean, var], num), ...}

    Return: [gene_num, 1024], [gene_num]
    """
    p_sample_list = []
    p_label_list = []
    class_num = len(CatEmbDict_avg.keys())
    for k, v in CatEmbDict_avg.items():
        mean = v[0].to('cuda')
        var = v[1].to('cuda')
        std_dev = torch.sqrt(var)
        pseudo_samples = [torch.normal(mean, std_dev) for _ in range(int(gene_num/class_num+1))]  # [num, 1024]
        pseudo_samples = torch.stack(pseudo_samples, dim=0)  # [num, 1024]
        pseudo_labels = torch.full((int(gene_num/class_num+1),), k)  # [num,]

        p_sample_list.append(pseudo_samples)  # [[num,1024], [num, 1024], ...]
        p_label_list.append(pseudo_labels)  # [[num,], [num, ], ...]
    
    p_sample_cat = torch.cat(p_sample_list, dim=0)
    p_label_cat = torch.cat(p_label_list, dim=0)

    perm = torch.randperm(len(p_label_cat))
    perm = perm[:gene_num]

    p_sample_list = [p_sample_cat[i] for i in perm]
    p_label_list = [p_label_cat[i] for i in perm]

    p_sample_group = [torch.stack(p_sample_list[i:i+batch_size]) for i in range(0, len(p_sample_list), batch_size)]
    p_label_group = [torch.stack(p_label_list[i:i+batch_size]) for i in range(0, len(p_label_list), batch_size)]

    return p_sample_group, p_label_group