import os
from collections import OrderedDict
from typing import List, Optional, Tuple
from addict import Dict
import copy
import time
import logging
from logging import debug, info


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10, CIFAR100

import flwr as fl
from flwr.common import Metrics

import argparse
from utils.cfg_utils import read_yaml
from utils.fl_utils import FlowerClient, test, get_parameters, set_parameters, AverageMeter, SoftTarget, accuracy
from utils.log_utils import cus_logger
from network import define_tsnet
import wandb

def train(args, train_loader, nets, optimizer, criterions, epoch, distill_info, data_num):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	st_losses  = AverageMeter()
	h_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()
	
	cfg = args.cfg
	device = args.DEVICE
	logger1 = cus_logger(args, __name__)
	wandb.init(project=cfg["wandb_project"], name=cfg["logfile_info"], config=cfg)
	
	snet = nets['snet']
	tnet = nets['tnet']

	criterionST  = criterions['criterionST'].to(device)
	mseloss = nn.MSELoss()

	snet.train()
	criterionCls = torch.nn.CrossEntropyLoss().cuda()
	end = time.time()
	for i, (img, target) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)
		img = img.to(device)
		target = target.to(device)
		stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = snet(img)
		
		stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = [],[],[],[],[],[]
		for index, tnet_i in enumerate(tnet):
			_stem_t, _rb1_t, _rb2_t, _rb3_t, _feat_t, _out_t = tnet_i(img)
			# if out_t is None:
			# 	stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = _stem_t, _rb1_t, _rb2_t, _rb3_t, _feat_t, _out_t
			# else:
			# 	out_t =  _out_t
			stem_t.append(_stem_t[1])
			rb1_t.append(_rb1_t[1])
			rb2_t.append(_rb2_t[1])
			rb3_t.append(_rb3_t[1])
			feat_t.append(_feat_t)
			out_t.append(_out_t)
			# out_t.append(_out_t * data_num[index] / sum(data_num))
			# logger1.info(f'\n***** print details *****')
			# logger1.info(f'{data_num[index] / sum(data_num) = }')
			# logger1.info(f'{_out_t = }')

		out_t = torch.stack(out_t, dim=0)
		out_t = torch.sum(out_t, dim=0)
		stem_t = torch.stack(stem_t, dim=0)
		stem_t = torch.sum(stem_t, dim=0)
		rb1_t = torch.stack(rb1_t, dim=0)
		rb1_t = torch.sum(rb1_t, dim=0)
		rb2_t = torch.stack(rb2_t, dim=0)
		rb2_t = torch.sum(rb2_t, dim=0)
		rb3_t = torch.stack(rb3_t, dim=0)
		rb3_t = torch.sum(rb3_t, dim=0)
		feat_t = torch.stack(feat_t, dim=0)
		feat_t = torch.sum(feat_t, dim=0)

		if args.cfg.use_softmax_to_distill:
			out_s = F.softmax(out_s, dim=1)
			out_t = F.softmax(out_t, dim=1)
			st_loss = criterionST(out_s, out_t.detach())
		else:
			st_loss = criterionST(out_s, out_t.detach())

		h_loss = mseloss(rb1_s[1], rb1_t) + mseloss(rb2_s[1], rb2_t) \
					+ mseloss(rb3_s[1], rb3_t) + mseloss(feat_s, feat_t)
		loss = args.cfg.st_weight * st_loss + (1-args.cfg.st_weight) * h_loss
		# loss = st_loss

		# prec1, prec5 = accuracy(out_s, target, topk=(1,5))
		st_losses.update(st_loss.item(), img.size(0))
		h_losses.update(h_loss.item(), img.size(0))
		# top1.update(prec1.item(), img.size(0))
		# top5.update(prec5.item(), img.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if i % 100 == 0:
			log_str = f'Epoch[{epoch}]:[{i}/{len(train_loader)}] \t[{distill_info}] \t ST:{st_losses.val} \t Hidden:{h_losses.val} '
			logger1.info(log_str)
			wandb.log({f"[{distill_info}]ST": st_losses.val,
						 f"[{distill_info}]Hidden": h_losses.val})
	return nets
	

def distill_test(test_loader, nets):
	top1 = AverageMeter()
	top5 = AverageMeter()
	snet = nets['snet']
	snet.eval()
	for i, (img, target) in enumerate(test_loader, start=1):
		img = img.cuda(non_blocking=True)
		target = target.cuda(non_blocking=True)
		with torch.no_grad():
			stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = snet(img)
		prec1, prec5 = accuracy(out_s, target, topk=(1,5))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	return top1.avg, top5.avg




def adjust_lr(optimizer, epoch):
	scale   = 0.1
	lr_list =  [0.1] * 100
	lr_list += [0.1*scale] * 50
	lr_list += [0.1*scale*scale] * 50

	lr = lr_list[epoch-1]
	logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def ensemble_test(args, net_list, data_num, testloader):
	"""
	This ensemble test method use the most confident client model to be the final output,
	and ignore the other model's output.
	"""
	correct, total, loss = 0, 0, 0.0
	for net in net_list:
		net.eval()
	with torch.no_grad():
		for images, labels in testloader:
			images, labels = images.to(args.DEVICE), labels.to(args.DEVICE)
			#-------ljz
			# confidence_list = []
			# predicted_list = []
			outputs_list = []
			for index, net in enumerate(net_list):
				stem, rb1, rb2, rb3, feat, outputs = net(images)
				outputs = outputs.data
				outputs = F.softmax(outputs, dim=1)
				outputs_norm = torch.norm(outputs, dim=1, keepdim=True)
				# outputs_normalized = outputs / outputs_norm * (data_num[index] / sum(data_num))
				outputs_normalized = outputs / outputs_norm
				# print(f'{outputs_normalized.size() = }')
				outputs_list.append(outputs_normalized)
				# outputs = F.softmax(outputs_normalized*5, dim=1)
				# confidence, predicted = torch.max(outputs.data, 1)
				# confidence_list.append(confidence)
				# predicted_list.append(predicted)


			# confidence_list = torch.stack(confidence_list, dim=0)
			# predicted_list = torch.stack(predicted_list, dim=0)
			outputs_list = torch.stack(outputs_list, dim=0)
			# print(f'{outputs_list.size() = }')
			result_var = torch.var(outputs_list, dim=-1)
			# print(f'{result_var.size() = }')
			max_value, max_index = torch.max(result_var, dim=0)
			# print(f'{max_value.size() = }')
			# print(f'{max_index.size() = }')
			outputs_reduce = outputs_list[max_index, torch.arange(outputs_list.size(1)),:]
			# print(f'{outputs_reduce.size() = }')
			# outputs_reduce = torch.sum(outputs_list, dim=0)
			# pred_reduce = torch.sum(predicted_list, dim=0)

			conf, pred = torch.max(outputs_reduce, dim=-1)
			# info(f'{pred = }')
			# info(f'{labels = }')
			# info(f'{conf = }')
			# final_predict = []
			# for i in range(labels.size(0)):
			#     final_predict.append(predicted_list[max_conf[i], i])
			predicted = pred.to(args.DEVICE)
			#-------
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	loss /= len(testloader.dataset)
	accuracy = correct / total
	return loss, accuracy


if __name__ == '__main__':
	from data_utils import load_dataloader
	args = Dict()
	args.DEVICE = 'cpu'
	args.cfg.device = 'cpu'
	args.cfg.batch_size = 4
	_, testloader = load_dataloader(args, 'cifar10', '../../dataset', is_iid=1, dataloader_num=1)

	net1 = define_tsnet(args, name='resnet18', num_class=10)
	net2 = define_tsnet(args, name='resnet18', num_class=10)
	net_list = [net1, net2]

	loss, accuracy = ensemble_test(args, net_list, testloader)