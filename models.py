import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from config import *
from data import *


class CVRDD(nn.Module):
    def __init__(self, num_features, embedding_dim, bias_dim, mlp_dims, dropout, alpha, kl, fusion_mode):
        super(CVRDD, self).__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.bias_dim = bias_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.alpha = alpha
        self.kl = kl
        self.fusion_mode = fusion_mod

        # initialize embedding methods
        self.sizes = getattr(Data(), 'features_sizes')
        self.embedding = Embeds(self.sizes, self.embedding_dim)

        # initialize mlp layers
        self.base_embed_dim = getattr(self.embedding, 'base_embed_dim')
        self.bias_embed_dim = getattr(self.embedding, 'bias_embed_dim')
        self.mlp = MLPLayers(self.base_embed_dim, self.bias_embed_dim, self.bias_dim, self.mlp_dims, self.dropout)

        # initialize fc_out functions
        self.fc_out_base = nn.Linear(self.mlp_dims[-1], 1)
        self.fc_out_bias = nn.Linear(self.bias_dim, 1)

        # initialize fusion methods
        self.fusion = Fusion(self.bias_dim, alpha=self.alpha, fusion_mode=self.fusion_mode, activation=None)

    def forward(self, inputs):
        """calculate and fuse Ym, Ym* and Yd"""
        action, pre_embed = inputs

        embed_base, embed_bias = self.embedding(action, pre_embed)
        # [batch_size, 352], [batch_size, 64]

        hidden_base, hidden_bias = self.mlp((embed_base, embed_bias))
        # [batch_size, 352 -> 300 -> 200 -> 100], [batch_size, 64 -> 64]

        out_base = self.fc_out_base(hidden_base)
        out_bias = self.fc_out_bias(hidden_bias)
        # [batch_size, 1], [batch_size, 1]

        # fuse Ym, Ym* and Yd
        ym0d, ym1d = self.fusion((out_base, out_bias))  # ym0d = Ymd, ym1d = Ym*d
        # [batch_size, 1], [batch_size, 1]

        return ym0d.squeeze(), ym1d.squeeze(), nn.Sigmoid()(out_base)


class MLPLayers(nn.Module):
    def __init__(self, base_embed_dim, bias_embed_dim, bias_dim, mlp_dims, dropout):
        super(MLPLayers, self).__init__()

        self.base_embed_dim = base_embed_dim
        self.bias_embed_dim = bias_embed_dim
        self.bias_dim = bias_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout

        self.mlp_base = nn.Sequential(nn.Linear(self.base_embed_dim, self.mlp_dims[0], bias=True),
                                      nn.ReLU(),
                                      nn.Dropout(self.dropout[0]),
                                      nn.Linear(self.mlp_dims[0], self.mlp_dims[1], bias=True),
                                      nn.ReLU(),
                                      nn.Dropout(self.dropout[1]),
                                      nn.Linear(self.mlp_dims[1], self.mlp_dims[2], bias=True),
                                      nn.ReLU(),
                                      nn.Dropout(self.dropout[2])
                                      )

        self.fc_bias = nn.Linear(self.bias_embed_dim, self.bias_dim, bias=True)
        self.relu_bias = nn.ReLU()

        # self.mlp_bias = nn.Sequential(nn.Linear(self.bias_embed_dim, self.bias_dim, bias=True),
        #                               nn.ReLU())

    def forward(self, inputs):
        embed_base, embed_bias = inputs

        hidden_base = self.mlp_base(embed_base)
        # hidden_bias = self.mlp_bias(embed_bias)
        hidden_bias = self.fc_bias(embed_bias)
        hidden_bias = self.relu_bias(embed_bias + hidden_bias)

        return hidden_base, hidden_bias


class Embeds(nn.Module):
    def __init__(self, sizes, embedding_dim):
        super(Embeds, self).__init__()

        self.sizes = sizes
        self.embedding_dim = embedding_dim

        self.embedding0 = nn.Embedding(self.sizes[0], self.embedding_dim)
        self.embedding1 = nn.Embedding(self.sizes[1], self.embedding_dim)
        self.embedding2 = nn.Embedding(self.sizes[2], self.embedding_dim)
        self.embedding3 = nn.Embedding(self.sizes[3], int(self.embedding_dim / 2))
        self.embedding4 = nn.Embedding(self.sizes[4], self.embedding_dim)

        self.base_embed_dim = self.embedding_dim * 5 + int(self.embedding_dim / 2)
        self.bias_embed_dim = self.embedding_dim
        setattr(self, 'base_embed_dim', self.base_embed_dim)
        setattr(self, 'bias_embed_dim', self.bias_embed_dim)

        self.bn = nn.BatchNorm1d(self.base_embed_dim)

    def forward(self, action, pre_embed):
        """
        get embedding of user, device, item, author and duration ids
        transform id to one-hot to embeddings by learnable linear transformation
        """
        id_map = {'userid': 0, 'feedid': 1, 'duration_level': 2, 'device': 3, 'authorid': 4, 'pre': 5}

        user_embed = self.embedding0(action[:, 0])
        feed_embed = self.embedding1(action[:, 1])
        duration_embed = self.embedding2(action[:, 2])
        device_embed = self.embedding3(action[:, 3])
        author_embed = self.embedding4(action[:, 4])
        pre_embed = nn.Embedding.from_pretrained(pre_embed, freeze=True)(action[:, 1])  # 红豆泥？？？

        embed_base = torch.concatenate((user_embed, feed_embed, duration_embed, device_embed, author_embed, pre_embed), dim=-1)
        # embed_base = torch.concatenate((user_embed, feed_embed, duration_embed, device_embed, author_embed), dim=-1)
        embed_base = torch.squeeze(embed_base)
        embed_base = self.bn(embed_base.type(torch.bfloat16))
        embed_base = embed_base.float()

        embed_bias = torch.squeeze(duration_embed)

        # return user_embed, feed_embed, duration_embed, device_embed, author_embed, pre_embed
        return embed_base, embed_bias


class Fusion(nn.Module):
    def __init__(self, bias_dim, alpha, fusion_mode, activation=None, **kwargs):
        super(Fusion, self).__init__()

        self.bias_dim = bias_dim
        self.alpha = alpha
        self.fusion_mode = fusion_mode
        self.activation = activation

        self.kernel = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.log = torch.log

    def forward(self, inputs, training=False):
        o_base, o_bias = inputs

        base_logit = self._fusion_func(o_base, o_bias)  # Ym,d
        bias_logit = self._fusion_func(self.kernel, o_bias)  # Ym*,d

        # normalize(optional)
        base_output = self.sigmoid(base_logit)
        bias_output = self.sigmoid(bias_logit)
        # Since counterfactual inference takes the maximum prediction as the outcome,
        # whether or not to normalize 𝑌𝑚,𝑑 is optional.

        return base_output, bias_output

    def _fusion_func(self, o1, o2):
        global o_fusion
        self.eps = 1e-12
        if self.fusion_mode == 'sum':
            o_fusion = self.log(self.sigmoid(o1 + o2))
        elif self.fusion_mode == 'hm':
            y_hm = self.sigmoid(o1) * self.sigmoid(o2)
            o_fusion = self.log(y_hm / (y_hm + self.eps))
        elif self.fusion_mode == 'mp':
            o_fusion = o1 * self.sigmoid(o2)
        return o_fusion


class KLLoss(nn.Module):
    def __init__(self, kl):
        super(KLLoss, self).__init__()

        self.kl_loss = nn.KLDivLoss(reduction='mean')  # 计算所有元素的均值
        self.kl = kl
        self.log = torch.log

    def forward(self, inputs):
        base, bias = inputs
        # kl_loss = self.kl_loss(base, bias)

        # 原文的方法
        kl_loss = -base * self.log(bias)
        kl_loss = torch.mean(kl_loss)

        return kl_loss

