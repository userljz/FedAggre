import torch
from Utils.server_client_utils import EmbAE
from Utils.data_utils import load_dataloader_from_generate, load_dataloader
from addict import Dict
from collections import defaultdict

args = Dict()
args.cfg.dirichlet_alpha = 0.1
args.cfg.num_clients = 10
args.cfg.batch_size = 64
dataset_name = 'PathMNIST'
args.cfg.model_name = 'ViT-B/32'

if dataset_name == 'cifar100':
    train_img = torch.load('/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32Train_imgemb.pth')
    train_label = torch.load('/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32Train_labels.pth')
elif dataset_name == 'emnist':
    train_img = torch.load('/home/ljz/dataset/emnist_generated_vitb32/emnist_vitb32Train_imgemb.pth')
    train_label = torch.load('/home/ljz/dataset/emnist_generated_vitb32/emnist_vitb32Train_labels.pth')
elif dataset_name == 'PathMNIST':
    train_img = torch.load('/home/ljz/dataset/PathMNIST_generated_vitb32/PathMNIST_vitb32Train_imgemb.pth')
    train_label = torch.load('/home/ljz/dataset/PathMNIST_generated_vitb32/PathMNIST_vitb32Train_labels.pth')

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

# # ------ Load Model ------
# ae_model = EmbAE(512, 64).to('cuda')
# ae_model.load_state_dict(torch.load("/home/ljz/vae/ae_3000TrainData_01noise_state_dict.pth", map_location=torch.device('cuda')))

# gene_img_info_summary = {}
# for index, train_loader_i in enumerate(train_loaders):
#     img_dict = defaultdict(list, {})
#     for image, label in train_loader_i:
#         for i in range(len(label)):
#             image_i = image[i]
#             label_i = label[i]
#             img_dict[label_i.detach().cpu().item()].append(image_i.detach())

#     origin_img_dict = {}
#     for key, value in img_dict.items():
#         _img = torch.stack(value).to('cuda')
#         origin_img_dict[key] = _img.detach()
    
    
#     origin_img_info = {}
#     for k,v in img_dict.items():
#         _v = torch.stack(v).to('cuda')
#         o_m = _v.mean(dim=0)
#         o_var = _v.var(dim=0, unbiased=False)
#         origin_img_info[k] = [o_m, o_var, _v.size(0)]
    
#     gene_img_dict = {}
#     for key, value in img_dict.items():
#         _img = torch.stack(value).to('cuda')
#         _gene_img = ae_model.generate(_img)
#         gene_img_dict[key] = _gene_img.detach()


#     gene_img_info = {}
#     for key, value in gene_img_dict.items():
#         _mean = value.mean(dim=0)

#         _var = value.var(dim=0, unbiased=False)
#         gene_img_info[key] = [_mean, _var, value.size(0)]

#     for k,v in gene_img_info.items():
#         if k in gene_img_info_summary.keys():
#             mean1 = v[0].detach()
#             var1 = v[1].detach()
#             n1 = v[2]

#             mean2 = gene_img_info_summary[k][0].detach()
#             var2 = gene_img_info_summary[k][1].detach()
#             n2 = gene_img_info_summary[k][2]

#             combined_mean = (n1 * mean1 + n2 * mean2) / (n1 + n2)
#             combined_var = ((n1 * var1 + n2 * var2) + ((n1 * n2) / (n1 + n2)) * (mean1 - mean2) ** 2) / (n1 + n2)
            
#             gene_img_info_summary[k] = [combined_mean, combined_var, (n1+n2)]
#         else:

#             gene_img_info_summary[k] = [v[0].detach(), v[1].detach(), v[2]]

# torch.save(gene_img_info_summary, 'gene_img_info_summary.pth')
# print('Successfully save gene_img_info_summary.pth')
