import os
from collections import OrderedDict
from typing import List, Optional, Tuple
from addict import Dict
import copy
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
from utils.fl_utils import FlowerClient, test, get_parameters, set_parameters
from utils.data_utils import load_dataloader
from utils.log_utils import cus_logger

from network import define_tsnet
from custom_strategy import FedCustom




parser = argparse.ArgumentParser()
parser.add_argument('--yaml_name', type=str, default='basic.yaml', help='config file name')
parser.add_argument('--round', type=int, help='num of communication round between server and clients')
parser.add_argument('--num_clients', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--dirichlet_alpha', type=float)
parser.add_argument('--num_class', type=int)
parser.add_argument('--client_model', type=str)
parser.add_argument('--server_model', type=str)
parser.add_argument('--distill_lr', type=float)
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

# ------wandb config------
if args.cfg.wandb == 0:
    os.environ['WANDB_MODE'] = 'dryrun'

# ------log config------
log_name = f'{cfg.logfile_info}_{cfg.client_model}_{cfg.server_model}_{cfg.round}_{cfg.num_clients}_{cfg.client_dataset}'
args.log_name = log_name
logger1 = cus_logger(args, __name__, m='w')

# ------device config------
DEVICE = torch.device(args.cfg.device)
args.DEVICE = DEVICE

# ------start training------
logger1.info(f"-------------------------------------------Start a New------------------------------------------------")
logger1.info(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

print_dict(args)

# ------Initialize client model------
args.net = define_tsnet(args, args.cfg.client_model, args.cfg.num_class)
client_resources = args.cfg.client_resource

# ------Initialize dataloader------
# trainloaders = load_dataloader(args, cfg.client_dataset, cfg.dataset_path, is_iid=0, dataloader_num=args.cfg.num_clients)
# _, valloaders, testloader = load_dataloader(args, cfg.client_dataset, cfg.dataset_path, is_iid=1, dataloader_num=args.cfg.num_clients)
trainloaders, valloaders = load_dataloader(args, cfg.client_dataset, cfg.dataset_path, is_iid=0, dataloader_num=args.cfg.num_clients)
_, _, testloader = load_dataloader(args, cfg.client_dataset, cfg.dataset_path, is_iid=1, dataloader_num=args.cfg.num_clients)

trainloader_d, _ = load_dataloader(args, cfg.server_dataset, cfg.dataset_path, is_iid=1, dataloader_num=1) # _d means distillation
_, testloader_d = load_dataloader(args, cfg.client_dataset, cfg.dataset_path, is_iid=1, dataloader_num=1) # _d means distillation

args.trainloaders, args.valloaders, args.testloader, args.trainloader_d, args.testloader_d = trainloaders, valloaders, testloader, trainloader_d, testloader_d

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    _trainloader = trainloaders[int(cid)]
    _valloader = valloaders[int(cid)]
    return FlowerClient(cid, args.net, _trainloader, _valloader, args)



fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=args.cfg.num_clients,
    config=fl.server.ServerConfig(num_rounds=args.cfg.round),
    strategy=FedCustom(args=args),  # <-- pass the new strategy here
    client_resources=client_resources,
)


logger1.info(f"-------------------------------------------End All------------------------------------------------")

# on branch v1