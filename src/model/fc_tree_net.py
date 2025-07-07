import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCTreeNet(torch.nn.Module):
    def __init__(self, in_dim=300, img_dim=256):
        '''
        initialization for TreeNet model, basically a ChildSumLSTM model
        with non-linear activation embedding for different nodes in the AoG.
        Shared weigths for all LSTM cells.
        :param in_dim:      input feature dimension for word embedding (from string to vector space)
        :param img_dim:     dimension of the input image feature, should be (panel_pair_number * img_feature_dim (e.g. 512 or 256))
        '''
        super(FCTreeNet, self).__init__()
        self.in_dim = in_dim
        self.img_dim = img_dim
        self.fc = nn.Linear(self.in_dim, self.in_dim)
        self.leaf = nn.Linear(self.in_dim + self.img_dim, self.img_dim)
        self.middle = nn.Linear(self.in_dim + self.img_dim, self.img_dim)
        self.merge = nn.Linear(self.in_dim + self.img_dim, self.img_dim)
        self.root = nn.Linear(self.in_dim + self.img_dim, self.img_dim)

        self.relu = nn.ReLU()

    def forward(self, image_feature, input, indicator):
        '''
        Forward function for TreeNet model
        :param input:		input should be (batch_size * 6 * input_word_embedding_dimension), got from the embedding vector
        :param indicator:	indicating whether the input is of structure with branches (batch_size * 1)
        :param image_feature:   input dictionary for each node, primarily feature, for example (batch_size * 16 (panel_pair_number) * feature_dim (output from CNN))
        :return:
        '''
        # Input validation
        if image_feature is None or input is None or indicator is None:
            raise ValueError("All inputs must be provided")
        
        # Ensure inputs are on the same device
        device = image_feature.device
        input = input.to(device)
        indicator = indicator.to(device)
        
        # Validate input shapes
        if input.dim() != 3 or input.size(-1) != self.in_dim:
            raise ValueError(f"Input should have shape (batch_size, 6, {self.in_dim}), got {input.shape}")
        
        if indicator.dim() != 2 or indicator.size(-1) != 1:
            raise ValueError(f"Indicator should have shape (batch_size, 1), got {indicator.shape}")
        
        if image_feature.dim() != 3 or image_feature.size(-1) != self.img_dim:
            raise ValueError(f"Image feature should have shape (batch_size, panel_pairs, {self.img_dim}), got {image_feature.shape}")
        
        # Process input through FC layer
        input = self.fc(input.view(-1, input.size(-1)))
        input = input.view(-1, 6, input.size(-1))
        input = input.unsqueeze(1).repeat(1, image_feature.size(1), 1, 1)
        indicator = indicator.unsqueeze(1).repeat(1, image_feature.size(1), 1).view(-1, 1)

        # Extract different node types
        leaf_left = input[:, :, 3, :].view(-1, input.size(-1))           # (batch_size * panel_pair_num) * input_word_embedding_dimension
        leaf_right = input[:, :, 5, :].view(-1, input.size(-1))
        inter_left = input[:, :, 2, :].view(-1, input.size(-1))
        inter_right = input[:, :, 4, :].view(-1, input.size(-1))
        merge = input[:, :, 1, :].view(-1, input.size(-1))
        root = input[:, :, 0, :].view(-1, input.size(-1))
        
        # Concatenate image_feature and word_embeddings for leaf node inputs
        image_feature_flat = image_feature.view(-1, image_feature.size(-1))
        leaf_left = torch.cat((leaf_left, image_feature_flat), dim=-1)
        leaf_right = torch.cat((leaf_right, image_feature_flat), dim=-1)

        # Process leaf nodes
        out_leaf_left = self.leaf(leaf_left)
        out_leaf_right = self.leaf(leaf_right)
        out_leaf_left = self.relu(out_leaf_left)
        out_leaf_right = self.relu(out_leaf_right)

        # Process intermediate nodes
        out_left = self.middle(torch.cat((inter_left, out_leaf_left), dim=-1))
        out_right = self.middle(torch.cat((inter_right, out_leaf_right), dim=-1))
        out_left = self.relu(out_left)
        out_right = self.relu(out_right)

        # Apply indicator mask and merge
        out_right = torch.mul(out_right, indicator)
        merge_input = torch.cat((merge, out_left + out_right), dim=-1)
        out_merge = self.merge(merge_input)
        out_merge = self.relu(out_merge)

        # Process root node
        out_root = self.root(torch.cat((root, out_merge), dim=-1))
        out_root = self.relu(out_root)
        
        # Return: size ((batch_size * panel_pair) * feature_dim)
        return out_root
        