import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from basic_model import BasicModel
from fc_tree_net import FCTreeNet

class conv_module(nn.Module):
    def __init__(self):
        super(conv_module, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x.view(-1, 32*4*4)

class mlp_module(nn.Module):
    def __init__(self):
        super(mlp_module, self).__init__()
        self.fc1 = nn.Linear(32*4*4, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 8)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args)
        self.conv = conv_module()
        self.mlp = mlp_module()
        self.fc_tree_net = FCTreeNet(in_dim=300, img_dim=512)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

    def compute_loss(self, output, target, meta_target, meta_structure):
        pred = output[0]
        loss = F.cross_entropy(pred, target)
        return loss

    def forward(self, x, embedding, indicator):
        alpha = 1.0
        if x.dim() == 4 and x.size(1) != 16:
            batch_size = x.size(0)
            x = x.view(batch_size, 16, 80, 80)
        
        features = self.conv(x)
        features_tree = features.view(-1, 1, 512)
        features_tree = self.fc_tree_net(features_tree, embedding, indicator)
        final_features = features + alpha * features_tree
        score = self.mlp(final_features)
        return score, None

    def load_model(self, path, epoch):
        try:
            state_dict = torch.load(path+'{}_epoch_{}.pth'.format(self.name, epoch), 
                                  map_location='cpu')['state_dict']
            self.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load model from {path}: {e}")
            raise e

    def save_model(self, path, epoch, acc, loss):
        try:
            torch.save({'state_dict': self.state_dict(), 'acc': acc, 'loss': loss}, 
                      path+'{}_epoch_{}.pth'.format(self.name, epoch))
        except Exception as e:
            print(f"Warning: Could not save model to {path}: {e}")
            raise e

class CNN_MLP_MIN(BasicModel):
    def __init__(self, args):
        super(CNN_MLP_MIN, self).__init__(args)
        self.conv = conv_module()
        self.mlp = mlp_module()
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

    def compute_loss(self, output, target, meta_target, meta_structure):
        pred = output[0]
        loss = F.cross_entropy(pred, target)
        return loss

    def forward(self, x, embedding, indicator):
        # Ensure input is properly shaped for the conv module
        if x.dim() == 4 and x.size(1) != 16:
            # If input is (batch, channels, height, width) but not 16 channels
            # Reshape to expected format
            batch_size = x.size(0)
            x = x.view(batch_size, 16, 80, 80)
        
        # Direct CNN -> MLP pipeline without tree network
        features = self.conv(x)
        score = self.mlp(features)
        return score, None

    def load_model(self, path, epoch):
        try:
            state_dict = torch.load(path+'{}_epoch_{}.pth'.format(self.name, epoch), 
                                  map_location='cpu')['state_dict']
            self.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load model from {path}: {e}")
            raise e

    def save_model(self, path, epoch, acc, loss):
        try:
            torch.save({'state_dict': self.state_dict(), 'acc': acc, 'loss': loss}, 
                      path+'{}_epoch_{}.pth'.format(self.name, epoch))
        except Exception as e:
            print(f"Warning: Could not save model to {path}: {e}")
            raise e

    