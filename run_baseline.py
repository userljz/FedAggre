from addict import Dict
import torch
import torch.nn as nn
import argparse
from collections import defaultdict
from network import Baseline_from_generated, Baseline_from_timm
import wandb
import copy

from Utils.cfg_utils import read_yaml
from Utils.data_utils import load_dataloader_from_generate, load_dataloader
from Utils.log_utils import cus_logger
from Utils.typing_utils import FitRes, FitIns
from Utils.server_client_utils import SupConLoss, train_baseline, test_baseline, get_parameters, set_parameters, CategoryEmbedding
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


# ------wandb config------
if args.cfg.wandb:
    wandb.init(project=args.cfg["wandb_project"], name=args.cfg["logfile_info"], config=args.cfg)


# ------ Number Classes Config ------
if args.cfg.client_dataset == 'cifar100':
    args.cfg.num_class = 100
elif args.cfg.client_dataset == 'emnist':
    args.cfg.num_class = 47
elif args.cfg.client_dataset == 'PathMNIST':
    args.cfg.num_class = 9
elif args.cfg.client_dataset == 'emnist62':
    args.cfg.num_class = 62
elif args.cfg.client_dataset == 'OrganAMNIST':
    args.cfg.num_class = 11
else:
    print('Please specify the num_class')

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
# text_emb = torch.load(f'/home/ljz/dataset/cifar100_classn_text/cifar100_classn_RN50_textemb.pth')

# if args.cfg.reverse_textemb == 1:
#     text_emb = text_emb.flip(dims=[0])

# args.text_emb = text_emb.float().to(args.cfg.device)


# ------ Initialize class embedding collector ------
# args.catemb = CategoryEmbedding()

# ------ Initialize DataLoader & Server ------
if args.cfg.use_timm == 1:
    train_loaders, test_loader = load_dataloader_from_generate(args, args.cfg.client_dataset, dataloader_num=args.cfg.num_clients)
else:
    train_loaders, test_loader = load_dataloader_from_generate(args, args.cfg.client_dataset, dataloader_num=args.cfg.num_clients)


def client_fn(cid, _args):
    if _args.cfg.use_timm == 1:
        _model = Baseline_from_timm(_args)
    else:
        _model = Baseline_from_generated(_args)
    
    _train_loader = train_loaders[cid]
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
    def __init__(self, args):
        self.args = args
        

    def aggregate_fit(self, server_round, results):
        weights_results = [
            (fit_res.parameters, fit_res.num_examples)
            for fit_res in results
        ]

        parameters_aggregated = aggregate(weights_results)
        return parameters_aggregated

    def configure_fit(self, server_round, parameters):
        config = {'server_round': server_round}

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
        # self.criterion = SupConLoss(self.args, self.args.text_emb)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)

        optimizer = torch.optim.SGD(self.net.parameters(),
                                    lr=float(self.args.cfg.local_lr),
                                    momentum=float(self.args.cfg.local_momentum),
                                    weight_decay=float(self.args.cfg.local_weight_decay))


        # ------ train multiple epochs ------
        if self.args.cfg.strategy == 'FedProx':
            global_model = copy.deepcopy(self.net)
        else:
            global_model = None
        
        for epoch in range(1, self.args.cfg.local_epoch + 1):
            client_loss = train_baseline(self.args, self.train_loader, self.net, self.criterion, optimizer, epoch, self.cid, config, global_model=global_model)
            logger1.info(f"Round[{config['server_round']}]TrainClient{self.cid}:[{epoch}/{self.args.cfg.local_epoch}], Loss:{client_loss:.3f}, DataLength:{len(self.train_loader.dataset)}")
            if self.args.cfg.wandb: wandb.log({f"Train_Client{self.cid}|Train_loss:": client_loss})

        return FitRes(get_parameters(self.net), len(self.train_loader), {})

    def evaluate(self, config):
        loss, _accuracy = test_baseline(self.args, self.net, self.test_loader, self.criterion)
        logger1.info(f"Round[{config['server_round']}]Val_Client[{self.cid}], Accu[{_accuracy:.3f}], Loss[{loss:.3f}]")
        if self.args.cfg.wandb: wandb.log({f"Val_Client{self.cid}|Accu:": _accuracy})
        return float(loss), float(_accuracy)  # ljz


# -------- Start FL ------
server = Server(args)
label_counts = count_labels(train_loaders)
for key, value in label_counts.items():
    logger1.info(f"[{key}]: {value}")

round_count = 0
while round_count < args.cfg.round:
    round_count += 1
    logger1.info(f'*** Round [{round_count}] ***')
    # ------ Config Round ------
    if round_count == 1:
        param = get_parameters(Baseline_from_generated(args))
        fit_ins = FitIns(param, {'server_round': 1})
    else:
        fit_ins.config['server_round'] = round_count

    # ------ Train Clients ------
    results = []
    accu_list, loss_list = [], []

    for client_id in range(args.cfg.num_clients):
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
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        aggregate_net = Baseline_from_generated(args)
        aggregate_net = set_parameters(aggregate_net, fit_ins.parameters)
        loss, accu = test_baseline(args, aggregate_net, test_loader, criterion)
    logger1.info(f'*** Round[{round_count}]: Server_Test_Accu[{accu:.3f}] *** ')
    if args.cfg.wandb: wandb.log({f"Server Test Accu": accu, 'epoch': round_count})


logger1.info(f"-------------------------------------------End All------------------------------------------------")