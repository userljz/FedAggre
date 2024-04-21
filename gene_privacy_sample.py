import torch
from Utils.server_client_utils import EmbAE
from Utils.data_utils import load_dataloader_from_generate, load_dataloader
from Utils.mean_estimation import PRIME
from addict import Dict
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def estimate_private_mean(img_emb_stack):
    _mean = img_emb_stack.mean(dim=0).detach().cpu().numpy()
    _var = img_emb_stack.var(dim=0, unbiased=False).detach().cpu().numpy()
    print(f"{_mean.shape = }")  # _mean.shape = (512,)
    print(f"{_var.shape = }")  # _var.shape = (512,)
    
    origin_img_emb = img_emb_stack.detach().cpu().numpy()
    X_good = np.random.normal(_mean, _var, size=(int(100 * 0.95), len(_mean)))
    X_bad = np.random.normal(_mean, _var, size=(int(100 * 0.05), len(_mean)))
    bad_mean = np.mean(X_bad, axis=0)
    diff = np.mean(np.abs(bad_mean - _mean) / np.abs(_mean))
    print(f"bad_mean {diff = }")
    X = np.concatenate([X_good, X_bad], axis=0)
    X = np.concatenate([X, origin_img_emb], axis=0)

    mean_list = []
    for trial in tqdm(range(10)):
        mean = PRIME(epsilon=10, delta=0.01, X=X, alpha=0.05, R=10)
        if mean is None:
            print("mean is None")
            continue
        else:
            mean_list.append(mean)
        
        # diff = np.sum(np.abs(real_mean - mean))
        # print(f"{diff = }")
    
    stacked_arr = np.stack(mean_list)
    mean = np.mean(stacked_arr, axis=0)
    # diff = np.sum(np.abs(real_mean - mean))
    # print(f"{diff = }")
    mean = torch.from_numpy(mean)

    return mean



if __name__ == "__main__":
    dataset_name = 'cifar100-ViT-B32-timm'
    private_mean = False
    is_save = True

    if dataset_name == 'cifar100':
        train_img = torch.load('/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32Train_imgemb.pth')
        train_label = torch.load('/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32Train_labels.pth')
    elif dataset_name == 'emnist' or dataset_name == 'emnist62':
        train_img = torch.load(f'/home/ljz/dataset/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Train_imgemb.pth')
        train_label = torch.load(f'/home/ljz/dataset/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Train_labels.pth')
    elif dataset_name == 'PathMNIST' or dataset_name == 'OrganAMNIST':
        train_img = torch.load(f'/home/ljz/dataset/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Train_imgemb.pth')
        train_label = torch.load(f'/home/ljz/dataset/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Train_labels.pth')
    elif dataset_name == 'cifar100-RN50':
        train_img = torch.load('/home/ljz/dataset/cifar100_generated/cifar100Train_RN50_imgembV1.pth')
        train_label = torch.load('/home/ljz/dataset/cifar100_generated/cifar100Train_labelsV1.pth')
    elif dataset_name == 'cifar100-ViT-B32-timm':
        train_img = torch.load(f'/home/ljz/dataset/cifar100_generated_vitb32/cifar100_ViT-B32-timmTrain_imgemb.pth')
        train_label = torch.load(f'/home/ljz/dataset/cifar100_generated_vitb32/cifar100_ViT-B32-timmTrain_labels.pth')
            
    elif dataset_name == 'cifar100-BLIP-base':
        train_img = torch.load(f'/home/ljz/dataset/cifar100_generated_BLIP/cifar100_BLIP-base_Train_imgemb.pth')
        train_label = torch.load('/home/ljz/dataset/cifar100_generated_BLIP/cifar100_BLIP-base_Train_labels.pth')
    elif dataset_name == 'cifar100-BLIP-base-noproj':
        train_img = torch.load(f'/home/ljz/dataset/cifar100_generated_BLIP/cifar100_BLIP-base-noproj_Train_imgemb.pth')
        train_label = torch.load('/home/ljz/dataset/cifar100_generated_BLIP/cifar100_BLIP-base-noproj_Train_labels.pth')
    elif dataset_name == 'cifar100-ALBEF-base-noproj':
        train_img = torch.load(f'/home/ljz/dataset/cifar100_generated_ALBEF/cifar100_ALBEF-base-noproj_Train_imgemb.pth')
        train_label = torch.load('/home/ljz/dataset/cifar100_generated_ALBEF/cifar100_ALBEF-base-noproj_Train_labels.pth')
    elif dataset_name == 'cifar100-ALBEF-base':
        train_img = torch.load(f'/home/ljz/dataset/cifar100_generated_ALBEF/cifar100_ALBEF-base_Train_imgemb.pth')
        train_label = torch.load('/home/ljz/dataset/cifar100_generated_ALBEF/cifar100_ALBEF-base_Train_labels.pth')
    else:
        print('Please specify the dataset')


    origin_img_dict = defaultdict(list, {})
    for index in range(len(train_label)):
        _indice = train_label[index].detach().cpu().item()
        _img = train_img[index].float().detach().cpu()
        origin_img_dict[_indice].append(_img)


    # ===== see different mean difference between different classes =====
    # mean_by_category = []
    # for k,v in origin_img_dict.items():
    #     _v = torch.stack(v)
    #     _mean = _v.mean(dim=0)
    #     mean_by_category.append(_mean)
    # mean_difference = []
    # for idx in range(len(mean_by_category)):
    #     mean_difference.append(torch.abs((mean_by_category[1]-mean_by_category[idx])/mean_by_category[1]).mean(dim=0))
    # print(f"Difference between means: {mean_difference}")
    # =================================================================


    origin_img_info = {}
    mean = []
    for k,v in origin_img_dict.items():
        _v = torch.stack(v)
        if private_mean:
            # _v = _v * 10
            # _mean = estimate_private_mean(_v)
            # real_mean = _v.mean(dim=0)
            # mean.append(real_mean)
            # diff = np.mean(np.abs(real_mean.numpy() - _mean.numpy()) / np.abs(real_mean.numpy()))
            # print(f"{k}: {diff}")
            real_mean = _v.mean(dim=0)
            _mean_noise = torch.normal(0, torch.abs(real_mean)*0.07)
            _mean = real_mean + _mean_noise
        else:
            _mean = _v.mean(dim=0)
        _var = _v.var(dim=0, unbiased=False)
        origin_img_info[k] = [_mean, _var, _v.size(0)]

    if is_save and private_mean:
        torch.save(origin_img_info, f'{dataset_name}_private_img_info.pth')
        print(f'Successfully save {dataset_name}_private_img_info.pth')
    elif is_save:
        torch.save(origin_img_info, f'{dataset_name}_origin_img_info.pth')
        print(f'Successfully save {dataset_name}_origin_img_info.pth')

