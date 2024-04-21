import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm
import copy
from Utils.server_client_utils import SupConLoss, test, set_parameters, test_baseline
from network import ClipModel_from_generated, Baseline_from_generated
from addict import Dict
from collections import defaultdict
from Utils.data_utils import load_dataloader_from_generate
import torch.nn as nn

import loss_landscapes
import loss_landscapes.metrics

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

# 参数的原始形状
canal_shapes = [
    torch.Size([2048, 512]),  # model.0.weight
    torch.Size([2048]),       # model.0.bias
    torch.Size([512, 2048]),  # model.2.weight
    torch.Size([512])         # model.2.bias
]

fedavg_shapes = [
    torch.Size([2048, 512]),  # model.0.weight
    torch.Size([2048]),       # model.0.bias
    torch.Size([100, 2048]),  # model.2.weight
    torch.Size([100])         # model.2.bias
]
def flatten_2_model(flattened_params, shapes):
    """
    flattened_params: a tensor, e.g. torch.Size([2099712]) 
    shapes: fedavg_shapes = [
                            torch.Size([2048, 512]),  # model.0.weight
                            torch.Size([2048]),       # model.0.bias
                            torch.Size([100, 2048]),  # model.2.weight
                            torch.Size([100])         # model.2.bias
                        ]
    """
    # 计算每个参数展平后的长度
    sizes = [torch.prod(torch.tensor(shape)).item() for shape in shapes]

    # 用于存储恢复后参数的列表
    params = []

    # 累积总长度，用于展平参数的索引
    offset = 0
    for shape, size in zip(shapes, sizes):
        # 提取展平参数中对应的部分
        param_flat = flattened_params[offset:offset + size]
        
        # 重塑为原始形状
        param = param_flat.view(shape)
        
        # 存储到列表中
        params.append(param)
        
        # 更新索引
        offset += size
    
    return params


def get_image_feat(test_img_label_list, model):
    image_feat_dict = defaultdict(list)
    for i in tqdm(test_img_label_list):
        img, label = i
        img = img.to('cuda')
        label = label.item()
        _feat = model(img.unsqueeze(0))
        image_feat_dict[label].append(_feat)
    return image_feat_dict



fedavg_001_10x40 = torch.load('/home/ljz/FedAggre/FedAvg_ClientParam_loss_landscope_alpha0.01_10clients.pth')
canal_001_10x40 = torch.load('/home/ljz/FedAggre/CANAL_ClientParam_loss_landscope_alpha0.01_10clients.pth')

test_img = torch.load('/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32Test_imgemb.pth')
test_label = torch.load('/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32Test_labels.pth')
test_img = test_img.float()
test_img_label_list = [(test_img[i], test_label[i]) for i in range(len(test_label))]

print(test_img_label_list[0][0].size())

args = Dict()
args.cfg.model_name = 'ViT-B/32'
args.cfg.use_mlp = 1
args.cfg.num_class = 100
device = torch.device('cuda')
args.device = device
args.cfg.margin = 0.2
# text_emb = text_emb.float().to(device)
# args.text_emb = text_emb
args.cfg.client_dataset = 'cifar100'
args.cfg.only_test_training_labels = 0
args.cfg.batch_size = 128
args.cfg.mlp_hiddenlayer_num = 2048

args.stragegy = 'fedavg'
use_epoch = 10
use_client = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if args.stragegy == 'canal':
    args.model = ClipModel_from_generated(args)
    param = canal_001_10x40
    shapes = canal_shapes
elif args.stragegy == 'fedavg':
    args.model = Baseline_from_generated(args)
    param = fedavg_001_10x40
    shapes = fedavg_shapes



# ============================================================================ #
# save img_feat_all_client of specific epoch
# ============================================================================ #
if os.path.isfile(f"/home/ljz/FedAggre/0129vis_feat/{args.stragegy}Round{use_epoch}_img_feat_all_client.pth"):
    print(f"The file exists")
else:
    img_feat_all_client = []
    """
    img_feat_all_client = [dict1, dict2, ..., dict10]  # because there are 10 clients
    dict1 = {'1':[img_feat1, img_feat2, ..., img_feat99], 
            '2':[img_feat1, img_feat2, ..., img_feat99],
                ...
            '99':[img_feat1, img_feat2, ..., img_feat99]}  # because in CIFAR100 test set, there are 100 classes, and 100 samples in each class
    """

    for clienti in use_client:
        p = param[clienti][use_epoch]
        p = flatten_2_model(p, shapes)
        _model = set_parameters(args.model, p)

        image_feat_dict = get_image_feat(test_img_label_list, _model)
        img_feat_all_client.append(image_feat_dict)

    torch.save(img_feat_all_client, f"/home/ljz/FedAggre/0129vis_feat/{args.stragegy}Round{use_epoch}_img_feat_all_client.pth")

# ============================================================================ #

img_feat_all_client = torch.load(f"/home/ljz/FedAggre/0129vis_feat/{args.stragegy}Round{use_epoch}_img_feat_all_client.pth")
for clienti in img_feat_all_client:
    for k,v in clienti.items():
        clienti[k] = [i.squeeze().detach().cpu().numpy() for i in v]



color = [
    'red', 'blue', 'green', 'cyan', 'magenta',
    'yellow', 'black', 'gray', 'orange',
    'pink', 'brown', 'gold', 'lime', 'olive',
    'maroon', 'navy', 'aqua', 'teal', 'fuchsia',
    'purple', 'silver'
]
markers = ['o', 's', '^','p', '*', '+', 'x', 'D', 'd', '|', '_']
color_list = []
marker_list = []
flatten_list = []
select_cat = 5

for idx, clienti in enumerate(img_feat_all_client):
    for cati in range(select_cat):
        flatten_list.extend(clienti[cati])
        marker_list.extend([markers[idx] for _ in range(len(clienti[cati]))])
        color_list.extend([color[cati] for _ in range(len(clienti[cati]))])

flatten_list = np.array(flatten_list)
print(flatten_list.shape)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=600)
tsne_results = tsne.fit_transform(flatten_list)

print(set(marker_list))
plt.figure(figsize=(30, 20))  # 创建一个宽10英寸、高5英寸的图
# 循环遍历坐标并逐一画出
for i,(x, y) in enumerate(tsne_results):
    plt.plot(x, y, marker=marker_list[i], color=color_list[i], markersize=8)  # 或者用plot绘制单个点，'o'表示点的形状
plt.savefig('/home/ljz/FedAggre/0129vis_feat/fedavg.png')