import torch
from Utils.server_client_utils import EmbAE
from Utils.data_utils import load_dataloader_from_generate, load_dataloader
from addict import Dict
from collections import defaultdict

dataset_name = 'emnist62'

if dataset_name == 'cifar100':
    train_img = torch.load('/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32Train_imgemb.pth')
    train_label = torch.load('/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32Train_labels.pth')
elif dataset_name == 'emnist' or dataset_name == 'emnist62':
    train_img = torch.load(f'/home/ljz/dataset/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Train_imgemb.pth')
    train_label = torch.load(f'/home/ljz/dataset/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Train_labels.pth')
elif dataset_name == 'PathMNIST' or dataset_name == 'OrganAMNIST':
    train_img = torch.load(f'/home/ljz/dataset/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Train_imgemb.pth')
    train_label = torch.load(f'/home/ljz/dataset/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Train_labels.pth')

else:
    print('Please specify the dataset')

# train_loaders, test_loader = load_dataloader_from_generate(args, dataset_name, dataloader_num=args.cfg.num_clients)

origin_img_dict = defaultdict(list, {})
for index in range(len(train_label)):
    _indice = train_label[index].detach().cpu().item()
    _img = train_img[index].float().detach().cpu()
    origin_img_dict[_indice].append(_img)

origin_img_info = {}
for k,v in origin_img_dict.items():
    _v = torch.stack(v)
    _mean = _v.mean(dim=0)
    _var = _v.var(dim=0, unbiased=False)
    origin_img_info[k] = [_mean, _var, _v.size(0)]

torch.save(origin_img_info, f'{dataset_name}_origin_img_info.pth')
print(f'Successfully save {dataset_name}_origin_img_info.pth')

