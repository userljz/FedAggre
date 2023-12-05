import os
from collections import OrderedDict
from typing import List, Optional, Tuple
from addict import Dict
import logging
from logging import debug, info, INFO
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, Subset

import flwr as fl
from flwr.common import Metrics

import argparse

from utils.cfg_utils import read_yaml, print_dict
from utils.fl_utils import FlowerClient, test, get_parameters, set_parameters, CategoryEmbedding
from utils.data_utils import load_dataloader_from_generate
from utils.log_utils import cus_logger

from network import ClipModel_from_generated
from custom_strategy import FedAvg_cus
import wandb



parser = argparse.ArgumentParser()
parser.add_argument('--yaml_name', type=str, default='basic.yaml', help='config file name')
parser.add_argument('--margin', type=float)
parser.add_argument('--use_mlp', type=int)
parser.add_argument('--local_lr', type=float)
parser.add_argument('--logfile_info', type=str)
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
wandb.init(project=args.cfg["wandb_project"], name=args.cfg["logfile_info"], config=args.cfg)
if args.cfg.wandb == 0:
    os.environ['WANDB_MODE'] = 'dryrun'


# ------log config------
log_name = f'{cfg.logfile_info}_{cfg.client_model}_{cfg.server_model}_{cfg.round}_{cfg.num_clients}_{cfg.client_dataset}'
args.log_name = log_name
logger1 = cus_logger(args, __name__, m='w')

# ------device config------
DEVICE = torch.device(args.cfg.device)
args.device = DEVICE

# ------start training------
logger1.info(f"-------------------------------------------Start a New------------------------------------------------")
logger1.info(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

print_dict(args)

# ------ Initialize CLIP text emb ------
text_emb = torch.load('/home/ljz/dataset/cifar10_generated/cifar10_RN50_textemb.pth')
args.text_emb = text_emb.float()


# ------ Initialize class embedding collector ------
args.catemb = CategoryEmbedding()
print('*** Init catemb ***')

# ------Initialize dataloader------
train_loaders, testloader = load_dataloader_from_generate(args, args.cfg.client_dataset, dataloader_num=args.cfg.num_clients)
args.trainloaders, args.testloader = train_loaders, testloader

# ------ Conduct some code only in test setting ------
if args.cfg.logfile_info == 'test':
    train_loaders, testloader = testloader, testloader


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    _trainloader = train_loaders[int(cid)]
    _testloader = testloader
    _model = ClipModel_from_generated(args)
    return FlowerClient(cid, _model, _trainloader, _testloader, args)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    # metrics is the return value of evaluate func of the FlowerClient
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    accu = sum(accuracies) / sum(examples)
    # logger1.info(f'*** in weighted_average ***')
    # logger1.info(f'{metrics=}')
    logger1.info(f'*** in weighted_average *** {accu=}')
    wandb.log({f"Weighted_average accuracy": accu})
    return {"accuracy": accu}

strategy = FedAvg_cus(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    evaluate_metrics_aggregation_fn=weighted_average,
    args = args
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=args.cfg.num_clients,
    config=fl.server.ServerConfig(num_rounds=args.cfg.round),
    client_resources=args.cfg.client_resource,
    strategy=strategy
)


logger1.info(f"-------------------------------------------End All------------------------------------------------")