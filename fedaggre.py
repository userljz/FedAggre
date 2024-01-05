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
parser.add_argument('--select_client_num', type=int)
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
    if args.cfg.model_name == 'RN50':
        text_emb = torch.load(f'/home/ljz/dataset/cifar100_classn_text/cifar100_classn_RN50_textemb.pth')
    elif args.cfg.model_name == 'ViT-B/32':
        text_emb = torch.load(f'/home/ljz/dataset/cifar100_generated_vitb32/cifar100_classn_vitb32_textemb.pth')
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
# gene_img_info_summary = {}
used_dict = {}
if args.cfg.use_before_fc_emb == 1:
    if args.cfg.client_dataset == 'cifar100':
        gene_img_info_summary = torch.load('gene_img_info_summary.pth')
        origin_img_info = torch.load('origin_img_info.pth')
    elif args.cfg.client_dataset == 'emnist':
        origin_img_info = torch.load('emnist_origin_img_info.pth')
    elif args.cfg.client_dataset == 'PathMNIST' or args.cfg.client_dataset == 'OrganAMNIST' or args.cfg.client_dataset == 'emnist62':
        origin_img_info = torch.load(f'{args.cfg.client_dataset}_origin_img_info.pth')
    else:
        print('Please specify the dataset')
    # for k,v in gene_img_info_summary.items():
    #     _mean = origin_img_info[k][0]
    #     _var = origin_img_info[k][1]
    #     used_dict[k] = [_mean, v[1], v[2]]

    used_dict = origin_img_info

    # ------ Load Model ------
    # ae_model = EmbAE(512, 64).to('cuda')
    # ae_model.load_state_dict(torch.load("/home/ljz/vae/ae_3000TrainData_01noise_state_dict.pth", map_location=torch.device('cuda')))

    # for index, train_loader_i in enumerate(train_loaders):
    #     img_dict = defaultdict(list, {})
    #     for image, label in train_loader_i:
    #         for i in range(len(label)):
    #             image_i = image[i]
    #             label_i = label[i]
    #             img_dict[label_i.detach().cpu().item()].append(image_i.detach())
        
    #     # origin_img_info = {}
    #     # for k,v in img_dict.items():
    #     #     _v = torch.stack(v).to('cuda')
    #     #     o_m = _v.mean(dim=0)
    #     #     o_var = _v.var(dim=0, unbiased=False)
    #     #     origin_img_info[k] = [o_m, o_var, _v.size(0)]
        
    #     gene_img_dict = {}
    #     # for key, value in img_dict.items():
    #     #     _img = torch.stack(value).to('cuda')
            
    #     #     if args.cfg.use_privacy_generator == 1:
    #     #         _gene_img = ae_model.generate(_img)
    #     #     else:
    #     #         _gene_img = _img

    #     #     gene_img_dict[key] = _gene_img.detach()
    #     # torch.save(gene_img_dict, 'gene_img_dict.pth')

    #     # for key, value in img_dict.items():
    #     #     _img = torch.stack(value).to('cuda')
            
    #     #     if args.cfg.use_privacy_generator == 1:
    #     #         _gene_img = ae_model.generate(_img)
    #     #     else:
    #     #         _gene_img = _img

    #     #     gene_img_dict[key] = _gene_img.detach()

    #     gene_img_dict = torch.load('gene_img_dict.pth')

    #     gene_img_info = {}
    #     for key, value in gene_img_dict.items():
    #         _mean = value.mean(dim=0)

    #         _var = value.var(dim=0, unbiased=False)
    #         # _var = origin_img_info[key][1]

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

    # print(gene_img_info_summary)




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
        if self.args.cfg.use_before_fc_emb == 1 and config['server_round'] > 1:
            CatEmbDict_avg = config['BeforeEmb_avg']
            gene_num = self.args.cfg.gene_num
            batch_size = self.args.cfg.batch_size
            p_images_group, p_labels_group = generate_from_meanvar(CatEmbDict_avg, gene_num, batch_size)
            logger1.info(f'Totally {len(p_images_group) * batch_size} Pseudo labels')
        
        # ------ train multiple epochs ------
        for epoch in range(1, self.args.cfg.local_epoch + 1):
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
while round_count < args.cfg.round:
    round_count += 1
    logger1.info(f'*** Round [{round_count}] ***')
    # ------ Config Round ------
    if round_count == 1:
        param = get_parameters(ClipModel_from_generated(args))
        fit_ins = FitIns(param, {'server_round': 1})
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


    for client_id in client_list_use:
        client = client_fn(client_id, args)

        # --- Fit One Client ---
        fit_res = client.fit(fit_ins.parameters, fit_ins.config)
        results.append(fit_res)

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

logger1.info(f"-------------------------------------------End All------------------------------------------------")