from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../CLIP')
import clip


class ClipModel(nn.Module):
	def __init__(self, args, model_name):
		super(ClipModel, self).__init__()
		self.model, _ = clip.load(model_name, device=args.device)
		if args.cfg.use_mlp == 1:
			self.fc_cus = nn.Sequential(
				nn.Linear(1024, 1024),
				nn.ReLU(),
				nn.Linear(1024, 1024)
			)
		else:
			self.fc_cus = nn.Linear(1024, 1024)
		self.args = args

	def get_img_feat(self, img):
		self.model, self.fc_cus = self.model.to(self.args.device), self.fc_cus.to(self.args.device)
		image_features = self.model.encode_image(img)
		image_features = image_features.to(torch.float32)

		image_features = self.fc_cus(image_features)
		return image_features

	def get_text_feat(self, text):
		# text = ["a diagram", "a dog", "a cat"]
		text = clip.tokenize(text)
		text = text.to(self.args.device)
		text_features = self.model.encode_text(text)
		return text_features

	def cus_train(self):
		for name, param in self.model.named_parameters():
			if "fc_cus" not in name:
				param.requires_grad = False





