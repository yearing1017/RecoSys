import datetime
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import TensorDataset, Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchkeras import summary, Model
from sklearn.metrics import auc, roc_auc_score, roc_curve

class Residual_block(nn.Module):
    def __init__(self, hidden_unit, dim_stack):
        super(Residual_block, self).__init__()
        self.linear1 = nn.Linear(dim_stack, hidden_unit)
        self.linear2 = nn.Linear(hidden_unit, dim_stack)
        self.relu = nn.ReLU()

    def forward(self, x):
        orign_x = x.clone()
        x = self.linear1(x)
        x = self.linear2(x)
        output = self.relu(x + orign_x)
        return output

class DeepCrossing(nn.Module):
    
    def __init__(self, feature_info, hidden_units, dropout=0., embed_dim=10, output_dim=1):
        """
        DeepCrossing：
            feature_info: 特征信息（数值特征， 类别特征， 类别特征embedding映射)
            hidden_units: 列表， 隐藏单元的个数(多层残差那里的)
            dropout: Dropout层的失活比例
            embed_dim: embedding维度
        """
        super(DeepCrossing, self).__init__()
        self.dense_feas, self.sparse_feas, self.sparse_feas_map = feature_info

        # embedding层， 这里需要一个列表的形式， 因为每个类别特征都需要embedding
        # key代表相应的稀疏特征列 val代表该列不同类别特征的个数；embd层对每列的特征进行embd
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=embed_dim)
            for key, val in self.sparse_feas_map.items()
        })

        # 统计embedding_dim的总维度：每个稀疏特征都需要一个embedding，个数乘以每个的dim
        embed_dim_sum = sum([embed_dim]*len(self.sparse_feas))

        # stack layers的总维度：数值特征的个数即该类的总维度 + emd层输出的总维度
        dim_stack = len(self.dense_feas) + embed_dim_sum

        # 残差层：
        self.res_layers = nn.ModuleList([
            Residual_block(unit, dim_stack) for unit in hidden_units
        ])

        # dropout层
        self.res_dropout = nn.Dropout(dropout)
        
        # 线性层
        self.linear = nn.Linear(dim_stack, output_dim)

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :13], x[:, 13:]
        sparse_inputs = sparse_inputs.long()      # 需要转成长张量， 这个是embedding的输入要求格式
        # 对每列的稀疏特征使用对应的embd层
        sparse_embd = [self.embed_layers['embed'+key](sparse_inputs[:,i]) 
                        for key,i in zip(self.sparse_feas_map.keys(), range(sparse_inputs.shape[1]))]
        # 先对稀疏的embd进行cat
        sparse_embed = torch.cat(sparse_embeds, axis=-1)
        stack = torch.cat([sparse_embed, dense_inputs], axis=-1)
        r = stack
        for res in self.res_layers:
            r = res(r)
        
        r = self.res_dropout(r)
        outputs = F.sigmoid(self.linear(r))
        return outputs

def get_Deep_Crossing(feature_info,hidden_unit):
    #hidden_units = [256, 128, 64, 32]
    net = DeepCrossing(feature_info, hidden_units)
    return net