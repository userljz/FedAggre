from addict import Dict
import torch
import argparse
from collections import defaultdict
from network import ClipModel_from_generated
import wandb
import copy
import numpy as np
import random

from Utils.cfg_utils import read_yaml
from Utils.data_utils import load_dataloader_from_generate, load_dataloader
from Utils.log_utils import cus_logger
from Utils.typing_utils import FitRes, FitIns
from Utils.server_client_utils import SupConLoss, train, test, get_parameters, set_parameters, CategoryEmbedding, generate_from_meanvar
from Utils.flwr_utils import aggregate
import math




parser = argparse.ArgumentParser()
parser.add_argument('--yaml_name', type=str, default='basic.yaml', help='config file name')
parser.add_argument('--dirichlet_alpha', type=float)
parser.add_argument('--use_before_fc_emb', type=int)
parser.add_argument('--local_lr', type=float)
parser.add_argument('--logfile_info', type=str)
parser.add_argument('--client_dataset', type=str)
parser.add_argument('--wandb_project', type=str)
parser.add_argument('--strategy', type=str)
parser.add_argument('--round', type=int)
parser.add_argument('--gene_num', type=int)
parser.add_argument('--num_clients', type=int)
parser.add_argument('--select_client_num', type=int)
parser.add_argument('--model_name', type=str)
parser.add_argument('--meaningful_anchor', type=int)
parser.add_argument('--theoretical_bound', type=int)
parser.add_argument('--save_model_param', type=int)
parser.add_argument('--save_client_param', type=int)
parser.add_argument('--local_epoch', type=int)
parser.add_argument('--fewshot_percentage', type=float)
parser.add_argument('--private_estimate', type=int)



args = parser.parse_args()
args_command = vars(args)

config_basic = read_yaml('basic.yaml')
cfg_basic = Dict(config_basic)
config_specific = read_yaml(args_command['yaml_name'])
cfg_basic.update(config_specific)
cfg = cfg_basic
args = Dict()
args.cfg = cfg
for k,v in args_command.items():
    if v is not None:
        args.cfg[k] = v

# ------ for test ------
if args.cfg.logfile_info == 'test':
    if args.cfg.use_softmax == 0:
        print('args.cfg.use_softmax == 0')
    print(type(args.cfg.use_softmax))





# ------ Number Classes Config ------
if args.cfg.client_dataset == 'cifar100':
    args.cfg.num_class = 100
elif args.cfg.client_dataset == 'emnist':
    args.cfg.num_class = 47
elif args.cfg.client_dataset == 'PathMNIST':
    args.cfg.num_class = 9
elif args.cfg.client_dataset == 'OrganAMNIST':
    args.cfg.num_class = 11
elif args.cfg.client_dataset == 'emnist62':
    args.cfg.num_class = 62
else:
    print('Please specify the num_class')

# ------wandb config------
if args.cfg.wandb:
    wandb.init(project=args.cfg["wandb_project"], name=args.cfg["logfile_info"], config=args.cfg)

# ------log config------
log_name = f'{cfg.wandb_project}_{cfg.logfile_info}_{cfg.client_dataset}_{cfg.round}_{cfg.num_clients}'
args.log_name = log_name
logger1 = cus_logger(args, __name__, m='w')


def print_dict(args):
    dict_info = copy.deepcopy(args)

    def info_dict(dict_in):
        for key, value in dict_in.items():
            if isinstance(value, dict):
                info_dict(value)
            else:
                logger1.info(f'{key} : {value}')
    
    info_dict(dict_info)


# ------start training------
args.device = args.cfg.device
logger1.info(f"-------------------------------------------Start a New------------------------------------------------")
logger1.info(f"Training on {args.cfg.device} using PyTorch {torch.__version__}")

print_dict(args)

# ------ Initialize CLIP text emb ------
# ============================================================================ #
# Use Traditional Text Emb
# text_emb = torch.load(f'/home/ljz/dataset/{args.cfg.client_dataset}_generated/{args.cfg.client_dataset}_RN50_textemb.pth')

# ============================================================================ #
# Use Category N text Emb
if args.cfg.client_dataset == 'cifar100':
    if args.cfg.meaningful_anchor == 1:
        # dim = 512
        if args.cfg.model_name == 'ViT-B/32' or args.cfg.model_name == 'ViT-B32-timm':
            text_emb = torch.load(f'/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32_textemb.pth')
    
    elif args.cfg.gpt3_anchor == "one_prompt":
        if args.cfg.model_name == 'ViT-B/32' or args.cfg.model_name == 'ViT-B32-timm':
            text_emb = torch.load(f'/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32_textemb_gpt3.pth')
    
    elif args.cfg.gpt3_anchor == "ensembel_prompt":
        if args.cfg.model_name == 'ViT-B/32' or args.cfg.model_name == 'ViT-B32-timm':
            text_emb = torch.load(f'/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32_textemb_gpt3Ensemble.pth')

    elif args.cfg.gpt3_anchor == "description":
        if args.cfg.model_name == 'ViT-B/32' or args.cfg.model_name == 'ViT-B32-timm':
            text_emb = torch.load(f'/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32_textemb_gpt3Descrip.pth')

    elif args.cfg.gpt3_anchor == "description_dim3072":
        text_emb = torch.load(f'/home/ljz/dataset/cifar100_generated_vitb32/cifar100_vitb32_textemb_gpt3Descrip_Dim3072.pth')

    
    else:
        if args.cfg.model_name == 'RN50':
            text_emb = torch.load(f'/home/ljz/dataset/cifar100_classn_text/cifar100_classn_RN50_textemb.pth')
        elif args.cfg.model_name == 'ViT-B/32' or args.cfg.model_name == 'ViT-B32-timm':
            text_emb = torch.load(f'/home/ljz/dataset/cifar100_generated_vitb32/cifar100_classn_vitb32_textemb.pth')
        elif args.cfg.model_name == 'BLIP-base' or args.cfg.model_name == 'BLIP-base-noproj':
            text_emb = torch.load(f'/home/ljz/dataset/cifar100_generated_BLIP/cifar100_classn_{args.cfg.model_name}_textemb.pth')
        elif args.cfg.model_name == 'ALBEF-base-noproj' or args.cfg.model_name == 'ALBEF-base':
            text_emb = torch.load(f'/home/ljz/dataset/cifar100_generated_ALBEF/cifar100_classn_{args.cfg.model_name}_textemb.pth')
elif args.cfg.client_dataset == 'emnist' or args.cfg.client_dataset == 'emnist62':
    if args.cfg.model_name == 'ViT-B/32':
        text_emb = torch.load(f'/home/ljz/dataset/{args.cfg.client_dataset}_generated_vitb32/{args.cfg.client_dataset}_classn_vitb32_textemb.pth')
        # text_emb = torch.load(f'/home/ljz/dataset/emnist_generated_vitb32/emnist_vitb32_textemb.pth')
elif args.cfg.client_dataset == 'PathMNIST' or args.cfg.client_dataset == 'OrganAMNIST':
    if args.cfg.model_name == 'ViT-B/32':
        text_emb = torch.load(f'/home/ljz/dataset/{args.cfg.client_dataset}_generated_vitb32/{args.cfg.client_dataset}_classn_vitb32_textemb.pth')

else:
    print('Please specify the dataset')

if args.cfg.reverse_textemb == 1:
    text_emb = text_emb.flip(dims=[0])

args.text_emb = text_emb.float().to(args.cfg.device)


# ------ Initialize DataLoader & Server ------
train_loaders, test_loader = load_dataloader_from_generate(args, args.cfg.client_dataset, dataloader_num=args.cfg.num_clients)

# ------ Train Each Client's Pseudo Sample Generator ------
used_dict = {}
if args.cfg.use_before_fc_emb == 1:
    if args.cfg.client_dataset == 'cifar100':
        if args.cfg.model_name == 'ViT-B/32':
            if args.cfg.private_estimate == 1:
                origin_img_info = torch.load('./ImgEmbStatistics/cifar100_private_img_info.pth')
            else:
                origin_img_info = torch.load('./ImgEmbStatistics/origin_img_info.pth')
        elif args.cfg.model_name == 'ViT-B32-timm':
            origin_img_info = torch.load('./ImgEmbStatistics/cifar100-ViT-B32-timm_origin_img_info.pth')
        elif args.cfg.model_name == 'RN50':
            origin_img_info = torch.load('./ImgEmbStatistics/cifar100-RN50_origin_img_info.pth')
        elif args.cfg.model_name == 'BLIP-base' or args.cfg.model_name == 'BLIP-base-noproj' or args.cfg.model_name == 'ALBEF-base-noproj' or args.cfg.model_name == 'ALBEF-base':
            origin_img_info = torch.load(f'./ImgEmbStatistics/cifar100-{args.cfg.model_name}_origin_img_info.pth')
    elif args.cfg.client_dataset == 'emnist':
        origin_img_info = torch.load('./ImgEmbStatistics/emnist_origin_img_info.pth')
    elif args.cfg.client_dataset == 'PathMNIST' or args.cfg.client_dataset == 'OrganAMNIST' or args.cfg.client_dataset == 'emnist62':
        origin_img_info = torch.load(f'./ImgEmbStatistics/{args.cfg.client_dataset}_origin_img_info.pth')
    else:
        print('Please specify the dataset')


    used_dict = origin_img_info

    


def client_fn(cid, _args):
    _model = ClipModel_from_generated(_args)
    _train_loader = train_loaders[cid]
    if args.cfg.only_test_training_labels == 1:
        _test_loader = test_loader[cid]
    else:
        _test_loader = test_loader
    return Client(_args, cid, _model, _train_loader, _test_loader)


def count_labels(data_loader):
    """
    In a non-iid setting, count the data distribution in each client's dataloader
    :param data_loader: [DataLoader1, DataLoader2, ...]
    :return: a Dict
    """
    label_counts = {}
    for client_indice, loader_i in enumerate(data_loader):
        _label_counts = {}
        _data_loader = loader_i
        for _, labels in _data_loader:
            # If labels are not already on CPU, move them
            labels = labels.cpu()
            for label in labels:
                # If label is tensor, convert to python number
                if isinstance(label, torch.Tensor):
                    label = label.item()
                # Increment the count for this label
                _label_counts[label] = _label_counts.get(label, 0) + 1
        _label_counts = dict(sorted(_label_counts.items()))
        label_counts[f'client{client_indice}'] = _label_counts

    return label_counts


class Server():
    def __init__(self, args, class_mean_var=None):
        self.args = args
        self.class_mean_var = class_mean_var

    def aggregate_fit(self, server_round, results):
        if results == None:
            print("In the first round, None results")
            return None
        
        weights_results = [
            (fit_res.parameters, fit_res.num_examples)
            for fit_res in results
        ]
        parameters_aggregated = aggregate(weights_results)
        return parameters_aggregated

    def configure_fit(self, server_round, parameters):
        config = {'server_round': server_round, 'BeforeEmb_avg': self.class_mean_var}
        fit_ins = FitIns(parameters, config)
        return fit_ins

    def server_conduct(self, server_round, results):
        parameters_aggregated = self.aggregate_fit(server_round, results)
        fit_ins = self.configure_fit(server_round, parameters_aggregated)
        return fit_ins


class Client():
    def __init__(self, args, cid, net, train_loader, test_loader):
        self.cid = cid
        self.net = net
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = SupConLoss(self.args, self.args.text_emb)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        optimizer = torch.optim.SGD(self.net.parameters(),
                                    lr=float(self.args.cfg.local_lr),
                                    momentum=float(self.args.cfg.local_momentum),
                                    weight_decay=float(self.args.cfg.local_weight_decay))


        # ------ Use Pseudo Emb ------
        p_images_group, p_labels_group = None, None
        if self.args.cfg.use_before_fc_emb == 1 and config['server_round'] > -1:
            CatEmbDict_avg = config['BeforeEmb_avg']
            gene_num = self.args.cfg.gene_num
            batch_size = self.args.cfg.batch_size
            p_images_group, p_labels_group = generate_from_meanvar(CatEmbDict_avg, gene_num, batch_size)
            logger1.info(f'Totally {len(p_images_group) * batch_size} Pseudo labels')
        
        # ------ train multiple epochs ------
        for epoch in range(1, self.args.cfg.local_epoch + 1):
            if args.cfg.theoretical_bound == 1:
                client_loss, batch_grads = train(self.args, self.train_loader, self.net, self.criterion,
                                    optimizer, epoch, self.cid, config, p_images_group, p_labels_group)
                batch_grad_mean = torch.mean(batch_grads, dim=0)  # torch.Size([2099712])
                batch_grad_var = torch.var(batch_grads, dim=0)
                mean_norm = torch.norm(batch_grad_mean, p=2).item()
                var_norm = torch.norm(batch_grad_var, p=2).item()
                cv = math.sqrt(var_norm) / mean_norm
                logger1.info(f"Round[{config['server_round']}]TrainClient{self.cid}:[{epoch}/{self.args.cfg.local_epoch}], Loss:{client_loss:.3f}, DataLength:{len(self.train_loader.dataset)}, mean/var:{mean_norm}/{var_norm}")
                if self.args.cfg.wandb: 
                    wandb.log({f"Train_Client{self.cid}|Train_loss:": client_loss})
                    wandb.log({f"Train_Client{self.cid}|Gradient_mean:": mean_norm})
                    wandb.log({f"Train_Client{self.cid}|Gradient_var:": var_norm})
                    wandb.log({f"Train_Client{self.cid}|CV:": cv})

            else:
                client_loss = train(self.args, self.train_loader, self.net, self.criterion,
                                    optimizer, epoch, self.cid, config, p_images_group, p_labels_group)
                logger1.info(f"Round[{config['server_round']}]TrainClient{self.cid}:[{epoch}/{self.args.cfg.local_epoch}], Loss:{client_loss:.3f}, DataLength:{len(self.train_loader.dataset)}")
                if self.args.cfg.wandb: wandb.log({f"Train_Client{self.cid}|Train_loss:": client_loss})

        return FitRes(get_parameters(self.net), len(self.train_loader), {})

    def evaluate(self, config):
        loss, _accuracy = test(self.args, self.net, self.test_loader, self.criterion)
        logger1.info(f"Round[{config['server_round']}]Val_Client[{self.cid}], Accu[{_accuracy:.3f}], Loss[{loss:.3f}]")
        if self.args.cfg.wandb: wandb.log({f"Val_Client{self.cid}|Accu:": _accuracy})
        return float(loss), float(_accuracy)  # ljz


# -------- Start FL ------
server = Server(args, used_dict)
label_counts = count_labels(train_loaders)
for key, value in label_counts.items():
    logger1.info(f"[{key}]: {value}")

round_count = 0
model_param = []
client_param = defaultdict(list)
while round_count < args.cfg.round:
    round_count += 1
    logger1.info(f'*** Round [{round_count}] ***')
    # ------ Config Round ------
    if round_count == 1:
        param = get_parameters(ClipModel_from_generated(args))
        # fit_ins = FitIns(param, {'server_round': 1})
        fit_ins = server.server_conduct(round_count, None)
        fit_ins.parameters = param
    else:
        fit_ins.config['server_round'] = round_count

    # ------ Train Clients ------
    results = []
    accu_list, loss_list = [], []
    
    # ------ Select Part of the Clients ------
    client_list = list(range(args.cfg.num_clients))
    if args.cfg.select_client_num == 0:
        client_list_use = client_list
    else:
        selected_client_numbers = random.sample(client_list, args.cfg.select_client_num)
        selected_client_numbers.sort()
        client_list_use = selected_client_numbers
    
    logger1.info(f'Round [{round_count}] select clients number is {client_list_use}')

    if args.cfg.save_model_param:
        _parameters_aggregated = fit_ins.parameters
        flattened_array = np.concatenate([p.ravel() for p in _parameters_aggregated])
        flattened_tensor = torch.from_numpy(flattened_array).clone()
        model_param.append(flattened_tensor)
        logger1.info(f"{len(model_param) = }")

    for client_id in client_list_use:
        client = client_fn(client_id, args)

        # --- Fit One Client ---
        fit_res = client.fit(fit_ins.parameters, fit_ins.config)
        results.append(fit_res)

        # --- Save One Client's Param ---
        if args.cfg.save_client_param:
            _client_parameters = fit_res.parameters
            client_flattened_array = np.concatenate([p.ravel() for p in _client_parameters])
            client_flattened_tensor = torch.from_numpy(client_flattened_array).clone()
            client_param[client_id].append(client_flattened_tensor)


        # --- Evaluate One Client
        loss, accu = client.evaluate(fit_ins.config)
        accu_list.append(accu)
        loss_list.append(loss)

    # ------ Server aggregate ------
    fit_ins = server.server_conduct(round_count, results)

    # ------ Test Aggregate Net ------
    if args.cfg.only_test_training_labels == 1:
         loss, accu = np.mean(loss_list), np.mean(accu_list)
    else:
        with torch.no_grad():
            criterion = SupConLoss(args, args.text_emb)
            aggregate_net = ClipModel_from_generated(args)
            aggregate_net = set_parameters(aggregate_net, fit_ins.parameters)
            loss, accu = test(args, aggregate_net, test_loader, criterion)
    
    logger1.info(f'*** Round[{round_count}]: Server_Test_Accu[{accu:.3f}] *** ')
    if args.cfg.wandb: wandb.log({f"Server Test Accu": accu, 'epoch': round_count})

if args.cfg.save_model_param:
    torch.save(model_param, f'{args.cfg.strategy}_loss_landscope_alpha{args.cfg.dirichlet_alpha}_{args.cfg.num_clients}clients.pth')
if args.cfg.save_client_param:
    torch.save(client_param, f'{args.cfg.strategy}_ClientParam_loss_landscope_alpha{args.cfg.dirichlet_alpha}_{args.cfg.num_clients}clients.pth')


logger1.info(f"-------------------------------------------End All------------------------------------------------")