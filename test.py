import torch
from Utils.data_utils import load_dataloader_from_generate
from addict import Dict
from collections import defaultdict
import torch.nn.functional as F

class_mean_var = torch.load('class_mean_var.pth')

# print(class_mean_var)

args = Dict()
args.cfg.dirichlet_alpha = 0.1
args.cfg.num_clients = 1
args.cfg.batch_size = 64
dataset_name = 'cifar100'
args.cfg.model_name = 'ViT-B/32'
train_loaders, test_loader = load_dataloader_from_generate(args, dataset_name, dataloader_num=args.cfg.num_clients)

origin_class_info = defaultdict(list)
for img, label in train_loaders[0]:
    for index in range(len(label)):
        origin_class_info[label[index].detach().cpu().item()].append(img[index].detach().cpu())

origin_class_meanvar = {}
for _key, _value in origin_class_info.items():
    _value = torch.stack(_value)
    _mean = _value.mean(dim=0)
    _var = _value.var(dim=0, unbiased=False)
    origin_class_meanvar[_key] = [_mean, _var]

# print(origin_class_meanvar)

for _k, _v in origin_class_meanvar.items():
    mean_p = class_mean_var[_k][0].cpu()
    var_p = class_mean_var[_k][1].cpu()

    mean_o = origin_class_meanvar[_k][0]
    var_o = origin_class_meanvar[_k][1]

    norm_mean_p = torch.norm(mean_p, p=2)
    norm_mean_o = torch.norm(mean_o, p=2)
    # print(f'Norm: {norm_mean_o= } | {norm_mean_p= }')

    similarity_mean = F.cosine_similarity(mean_p.unsqueeze(0), mean_o.unsqueeze(0), dim=1)
    l2_distances = F.pairwise_distance(mean_o*100, mean_p*100, p=2)
    # similarity_var = F.cosine_similarity(var_p.unsqueeze(0), var_o.unsqueeze(0), dim=1)

    # print(f'{_k}: Mean_Sim[{similarity_mean}]  l2_distances[{l2_distances}]')

for _k, _v in origin_class_meanvar.items():
    mean_p0 = class_mean_var[0][0].cpu()
    mean_p = class_mean_var[_k][0].cpu()

    mean_o1 = origin_class_meanvar[1][0]
    mean_o = origin_class_meanvar[_k][0]

    similarity_mean_p = F.cosine_similarity(mean_p.unsqueeze(0), mean_p0.unsqueeze(0), dim=1)
    similarity_mean_o = F.cosine_similarity(mean_o.unsqueeze(0), mean_o1.unsqueeze(0), dim=1)
    l2_distances_p = F.pairwise_distance(mean_p0*100, mean_p*100, p=2)
    l2_distances_o = F.pairwise_distance(mean_o*100, mean_o1*100, p=2)
    l2_distances_op = F.pairwise_distance(mean_o*100, mean_p*100, p=2)
    print(f'{_k}: \t MeanP_Dis[{l2_distances_p}] \t MeanO_Dis[{l2_distances_o}] \t OP_dist[{l2_distances_op}]')