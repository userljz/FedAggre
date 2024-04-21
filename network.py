from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '/home/ljz/CLIP')
import clip

import timm

class ClipModel(nn.Module):
    def __init__(self, args, model_name):
        super(ClipModel, self).__init__()
        self.model, _ = clip.load(model_name, device=args.device)
        if args.cfg.use_mlp == 1:
            self.fc_cus = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024)
            )
        else:
            self.fc_cus = nn.Linear(1024, 1024)
        self.args = args

    def get_img_feat(self, img, given_feat=None):
        """
        given_feat: Use the average of the features of other clients as a negative sample
        """
        if given_feat is None:
            self.model, self.fc_cus = self.model.to(self.args.device), self.fc_cus.to(self.args.device)
            image_features = self.model.encode_image(img)
            image_features = image_features.to(torch.float32)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        else:
            # print('*** Use Given Feat ***')
            image_features = img

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


class ClipModel_from_generated(nn.Module):
    def __init__(self, args):
        super(ClipModel_from_generated, self).__init__()
        
        if args.cfg.model_name == 'RN50':
            img_emb_length = 1024
            fc_input_dim = img_emb_length
            fc_output_dim = img_emb_length
        elif args.cfg.model_name == 'ViT-B/32':
            img_emb_length = 512
            fc_input_dim = img_emb_length
            fc_output_dim = img_emb_length
        elif args.cfg.model_name == 'ViT-B32-timm':
            img_emb_length = 1000
            fc_input_dim = img_emb_length
            fc_output_dim = 512
            
        elif args.cfg.model_name == 'BLIP-base' or args.cfg.model_name == 'ALBEF-base':
            img_emb_length = 256
            fc_input_dim = img_emb_length
            fc_output_dim = img_emb_length
        elif args.cfg.model_name == 'BLIP-base-noproj' or args.cfg.model_name == 'ALBEF-base-noproj':
            img_emb_length = 768
            fc_input_dim = img_emb_length
            fc_output_dim = img_emb_length
        else:
            print('Please specify the img_emb_length')

        if args.cfg.gpt3_anchor == "description_dim3072":
                fc_output_dim = 3072
        
        mlp_hiddenlayer_num = args.cfg.mlp_hiddenlayer_num
        
        if args.cfg.use_mlp == 1:
            self.model = nn.Sequential(
                nn.Linear(fc_input_dim, mlp_hiddenlayer_num),
                nn.ReLU(),
                nn.Linear(mlp_hiddenlayer_num, fc_output_dim)
            ).to(args.device)
        else:
            self.model = nn.Linear(fc_input_dim, fc_output_dim).to(args.device)
        self.args = args

    def get_img_feat(self, img_emb):
        """
        given_feat: Use the average of the features of other clients as a negative sample
        """
        image_features = self.model(img_emb)
        return image_features
    def forward(self, img_emb):
        image_features = self.model(img_emb)
        return image_features

class Baseline_from_generated(nn.Module):
    def __init__(self, args):
        super(Baseline_from_generated, self).__init__()

        if args.cfg.model_name == 'RN50':
            img_emb_length = 1024
        elif args.cfg.model_name == 'ViT-B/32':
            img_emb_length = 512
        elif args.cfg.model_name == 'ViT-B32-timm':
            img_emb_length = 1000
        else:
            print('Please specify the img_emb_length')

        mlp_hiddenlayer_num = args.cfg.mlp_hiddenlayer_num

        output_dim = int(args.cfg.num_class)
        if args.cfg.use_mlp == 1:
            self.model = nn.Sequential(
                nn.Linear(img_emb_length, mlp_hiddenlayer_num),
                nn.ReLU(),
                nn.Linear(mlp_hiddenlayer_num, output_dim)
            ).to(args.device)
        else:
            self.model = nn.Linear(img_emb_length, output_dim).to(args.device)
        self.args = args

    def forward(self, img_emb):
        """
        given_feat: Use the average of the features of other clients as a negative sample
        """
        image_features = self.model(img_emb)
        return image_features


class Baseline_from_timm(nn.Module):
    def __init__(self, args):
        super(Baseline_from_generated, self).__init__()
        output_dim = int(args.cfg.num_class)
        if args.cfg.model_name == 'RN50':
            self.model = timm.create_model('resnet50', num_classes=output_dim, pretrained=True)
        elif args.cfg.model_name == 'ViT-B/32':
            self.model = timm.create_model('vit_base_patch32_224.augreg_in21k', num_classes=output_dim, pretrained=True)
        else:
            print('Please specify the img_emb_length')

        self.args = args

    def forward(self, img):
        """
        img: [bs, 3, 224, 224]
        """
        image_features = self.model(img)
        return image_features

    def cus_train(self):
        if self.args.cfg.model_name == 'ViT-B/32':
            for name, param in self.model.named_parameters():
                if "head" not in name:
                    param.requires_grad = False

        elif self.args.cfg.model_name == 'RN50':
            for name, param in self.model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False


if __name__ == '__main__':
    # avail_pretrained_models = timm.list_models(pretrained=True)
    # for i in avail_pretrained_models:
    #     if 'vit_base' in i:
    #         print(i)

    model = timm.create_model('resnet50', num_classes=100, pretrained=True)
    # x = torch.randn(1, 3, 224, 224)
    # output_shape = model(x).shape
    # print(output_shape)
    for name, param in model.named_parameters():
        print(name)
        if "head" in name:
            print(param.shape)



