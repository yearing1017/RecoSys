import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationBasedAttention(nn.Module):
    def __init__(self, emb_dim, att_weight_dim):
        super(LocationBasedAttention, self).__init__()
        self.weights = nn.Parameter(torch.zeros(emb_dim, att_weight_dim))
        nn.init.xavier_uniform_(self.weights.data)
        self.bias = nn.Parameter(torch.randn(att_weight_dim))
        self.h = nn.Parameter(torch.randn(att_weight_dim))

    def forward(self, values):
        # values: N * num * emb_dim
        att_signal = torch.matmul(values, self.weights) # N * num * att_weight_dim
        att_signal = att_signal + self.bias # N * num * att_weight_dim
        att_signal = F.relu(att_signal)
        # mul为对应位广播相乘
        att_signal = torch.mul(att_signal, self.h) # N * num * att_weight_dim
        att_signal = torch.sum(att_signal, dim=2) # N * num ？为何要sum 
        att_signal = F.softmax(att_signal, dim=1) # N * num 
        return att_signal

class OutputLayer(nn.Module):
    def __init__(self, in_dim, out_type='binary', use_bias=True):
        super(OutputLayer, self).__init__()
        self.out_type = out_type
        self.in_dim = in_dim
        self.use_bias = use_bias
        if not self.in_dim == 1:
            self.weights = nn.Linear(in_features=in_dim, out_features=1, bias=self.use_bias)
        if self.out_type == 'binary':
            self.output_layer = nn.Sigmoid()

    def forward(self, x):
        if not self.in_dim == 1:
            y = self.weights(x)
        else:
            y = x
        if self.out_type == 'binary':
            y = self.output_layer(y)
        return y

def build_cross(num_fields, feat_emb):
    # num_pairs = num_fields * (num_fields-1) / 2
    row = []
    col = []
    for i in range(num_fields - 1):
        for j in range(i + 1, num_fields):
            row.append(i)
            col.append(j)
    p = feat_emb[:, row]  # N * num_pairs * emb_dim
    q = feat_emb[:, col]  # N * num_pairs * emb_dim
    return p, q

class AFM(nn.Module):
    def __init__(self, emb_dim, num_feats, num_fields, att_weight_dim, out_type='binary'):
        super(AFM, self).__init__()
        self.emb_dim = emb_dim
        self.num_feats = num_feats
        self.num_fields = num_fields
        self.att_weight_dim = att_weight_dim
        # 一阶权重
        self.first_order_weights = nn.Embedding(num_embeddings=num_feats, embedding_dim=1)
        nn.init.xavier_uniform_(self.first_order_weights.weight)
        self.bias = nn.Parameter(torch.randn(1))
        # embedding层
        self.emb_layer = nn.Embedding(num_embeddings=num_feats, embedding_dim=emb_dim)
        nn.init.xavier_uniform_(self.emb_layer.weight)
        # 成对组合的数量
        self.num_pairs = num_fields * (num_fields - 1) / 2
        # 注意力层
        self.att_layer = LocationBasedAttention(emb_dim, att_weight_dim)
        self.p = nn.Parameter(torch.randn(emb_dim))
        # 打分输出
        self.output_layer = OutputLayer(1, out_type)

    def forward(self, feat_index, feat_value):
        # feat_index和feat_value都是 N * num_fields * num_feats
        # unsqueeze表示在第3维增加一个维度，维度数为1
        feat_value = feat_value.unsqueeze(2) # N * num_fields * 1
        
        first_order_weight = self.first_order_weights(feat_index) # N * num_fields * 1
        y_first_order = torch.mul(first_order_weight, feat_value) # N * num_fields * 1
        y_first_order = torch.sum(y_first_order, dim=1) # N * 1,此处的加和对应公式中的求和
        y_first_order = y_first_order.squeeze(1) # 删除第2维

        feat_emb = self.emb_layer(feat_index) # N * num_fields * emb_dim
        feat_emb_value = torch.mul(feat_emb, feat_value) # N * num_fields * emb_dim
        p,q = build_cross(self.num_fields, feat_emb_value) 
        pair_wise_inter = torch.mul(p,q) # N * num_pairs * emb_dim 此处构建两两隐向量对应维度相乘

        att_signal = self.att_layer(pair_wise_inter)  # N * num_pairs
        att_signal = att_signal.unsqueeze(dim=2)  # N * num_pairs * 1

        att_inter = torch.mul(att_signal, pair_wise_inter)  # N * num_pairs * emb_dim
        att_pooling = torch.sum(att_inter, dim=1)  # N * emb_dim

        att_pooling = torch.mul(att_pooling, self.p)  # N * emb_dim
        att_pooling = torch.sum(att_pooling, dim=1)  # N

        y = self.bias + y_first_order + att_pooling
        y = self.output_layer(y)
        return y
