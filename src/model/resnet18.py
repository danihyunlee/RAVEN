import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from basic_model import BasicModel
from fc_tree_net import FCTreeNet

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()
    
    def forward(self, x):
        return x

class mlp_module(nn.Module):
    def __init__(self):
        super(mlp_module, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 8+9+21)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Resnet18_MLP(BasicModel):
    def __init__(self, args):
        super(Resnet18_MLP, self).__init__(args)
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = identity()
        self.mlp = mlp_module()
        self.fc_tree_net = FCTreeNet(in_dim=300, img_dim=512)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.meta_alpha = args.meta_alpha
        self.meta_beta = args.meta_beta

    def compute_loss(self, output, target, meta_target, meta_structure):
        pred, meta_target_pred, meta_struct_pred = output[0], output[1], output[2]

        target_loss = F.cross_entropy(pred, target)
        meta_target_pred = torch.chunk(meta_target_pred, chunks=9, dim=1)
        meta_target = torch.chunk(meta_target, chunks=9, dim=1)
        meta_target_loss = 0.
        for idx in range(0, 9):
            meta_target_loss += F.binary_cross_entropy(F.sigmoid(meta_target_pred[idx]), meta_target[idx])

        meta_struct_pred = torch.chunk(meta_struct_pred, chunks=21, dim=1)
        meta_structure = torch.chunk(meta_structure, chunks=21, dim=1)
        meta_struct_loss = 0.
        for idx in range(0, 21):
            meta_struct_loss += F.binary_cross_entropy(F.sigmoid(meta_struct_pred[idx]), meta_structure[idx])
        loss = target_loss + self.meta_alpha*meta_struct_loss/21. + self.meta_beta*meta_target_loss/9.
        return loss

    def forward(self, x, embedding, indicator):
        alpha = 1.0
        # Ensure input is properly shaped for ResNet18
        if x.dim() == 4 and x.size(1) != 16:
            # If input is (batch, channels, height, width) but not 16 channels
            # Reshape to expected format
            batch_size = x.size(0)
            x = x.view(batch_size, 16, 224, 224)
        
        features = self.resnet18(x)
        features_tree = features.view(-1, 1, 512)
        features_tree = self.fc_tree_net(features_tree, embedding, indicator)
        final_features = features + alpha * features_tree
        output = self.mlp(final_features)
        pred = output[:,0:8]
        meta_target_pred = output[:,8:17]
        meta_struct_pred = output[:,17:38]
        return pred, meta_target_pred, meta_struct_pred

    def load_model(self, path, epoch):
        """Override load_model to handle potential device issues"""
        try:
            state_dict = torch.load(path+'{}_epoch_{}.pth'.format(self.name, epoch), 
                                  map_location='cpu')['state_dict']
            self.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load model from {path}: {e}")
            raise e

    def save_model(self, path, epoch, acc, loss):
        """Override save_model to handle potential device issues"""
        try:
            torch.save({'state_dict': self.state_dict(), 'acc': acc, 'loss': loss}, 
                      path+'{}_epoch_{}.pth'.format(self.name, epoch))
        except Exception as e:
            print(f"Warning: Could not save model to {path}: {e}")
            raise e

class Resnet18_MLP_MIN(BasicModel):
    def __init__(self, args):
        super(Resnet18_MLP_MIN, self).__init__(args)
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = identity()
        self.mlp = mlp_module()
        # No fc_tree_net - removed for minimal version
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.meta_alpha = args.meta_alpha
        self.meta_beta = args.meta_beta

    def compute_loss(self, output, target, meta_target, meta_structure):
        pred = output[0]
        target_loss = F.cross_entropy(pred, target)
        return target_loss

    def forward(self, x, embedding, indicator):
        # Ensure input is properly shaped for ResNet18
        if x.dim() == 4 and x.size(1) != 16:
            # If input is (batch, channels, height, width) but not 16 channels
            # Reshape to expected format
            batch_size = x.size(0)
            x = x.view(batch_size, 16, 224, 224)
        
        # Direct ResNet18 -> MLP pipeline without tree network
        features = self.resnet18(x)
        output = self.mlp(features)
        pred = output[:,0:8]
        meta_target_pred = output[:,8:17]
        meta_struct_pred = output[:,17:38]
        return pred, meta_target_pred, meta_struct_pred

    def load_model(self, path, epoch):
        """Override load_model to handle potential device issues"""
        try:
            state_dict = torch.load(path+'{}_epoch_{}.pth'.format(self.name, epoch), 
                                  map_location='cpu')['state_dict']
            self.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load model from {path}: {e}")
            raise e

    def save_model(self, path, epoch, acc, loss):
        """Override save_model to handle potential device issues"""
        try:
            torch.save({'state_dict': self.state_dict(), 'acc': acc, 'loss': loss}, 
                      path+'{}_epoch_{}.pth'.format(self.name, epoch))
        except Exception as e:
            print(f"Warning: Could not save model to {path}: {e}")
            raise e
    