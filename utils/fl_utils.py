import os
from collections import OrderedDict
from typing import List, Optional, Tuple
from addict import Dict
from utils.log_utils import cus_logger



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




# class Net(nn.Module):
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

def train(args, net, cid, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    DEVICE = args.cfg.device
    custom_logger = cus_logger(args, __name__)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            stem, rb1, rb2, rb3, feat, outputs = net(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        custom_logger.info(f"Train_Client{cid}:[{epoch+1}/{epochs}], Train_loss:{epoch_loss}, Accu:{epoch_acc}, DataLength:{len(trainloader.dataset)}")


def test(args, net, testloader):
    """Evaluate the network on the entire test set."""
    # print('********** enter test in fl_utils **********')
    DEVICE = args.cfg.device
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            stem, rb1, rb2, rb3, feat, outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, args=None):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.args = args
        self.client_logger = cus_logger(args, 'client_logger')
        label_counts = self.count_labels(trainloader)
        self.client_logger.info(f'Cid_{self.cid}, label_counts:{label_counts}')

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.args, self.net, self.cid, self.trainloader, epochs=self.args.cfg.local_epoch, verbose=True)
        loss, accuracy = test(self.args, self.net, self.valloader)
        self.client_logger.info(f'Val_Client_{self.cid}, Accu:{accuracy}')
        if self.args.cfg.save_client_model == 1:
            torch.save(self.net, f"./save_client_model/client_{self.cid}.pth")
            self.client_logger.info(f'Finish saving client model {self.cid}')
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.client_logger.info('********** enter evaluate of FlowerClient **********')
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.args, self.net, self.valloader)
        self.client_logger.info(f'Cid:{self.cid}, Accu:{float(accuracy)}, Val_length:{len(self.valloader)}') # ljz
        return float(loss), len(self.valloader), {"accuracy": float(accuracy), 'test':12138} #ljz
    
    def count_labels(self, data_loader):
        # Initialize a counter dictionary
        label_counts = {}

        for _, labels in data_loader:
            # If labels are not already on CPU, move them
            labels = labels.cpu()
            for label in labels:
                # If label is tensor, convert to python number
                if isinstance(label, torch.Tensor):
                    label = label.item()
                # Increment the count for this label
                label_counts[label] = label_counts.get(label, 0) + 1

        return label_counts



# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]
#     print('metrics: ', metrics) # ljz
#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}



class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val   = 0
		self.avg   = 0
		self.sum   = 0
		self.count = 0

	def update(self, val, n=1):
		self.val   = val
		self.sum   += val * n
		self.count += n
		self.avg   = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred    = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss