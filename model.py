import math
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules import (BatchNorm1d, Dropout, Linear, MultiheadAttention,
                              TransformerEncoderLayer)


def build_encoder(cfg):

    model = TSTransformerEncoder(feat_dim = cfg.num_features, max_len = cfg.model_input_length, d_model = cfg.model_dim, n_heads = cfg.num_heads,
                               num_layers = cfg.num_layers, dim_feedforward = cfg.ff_dim, dropout= cfg.dropout_rate, norm = cfg.norm)
    
    model = model.apply(init_weights)

    model = model.to(cfg.device)

    return model

def build_clustering(cfg):
    model = nn.Sequential(
        nn.Linear(cfg.encoder_dim, cfg.clustering_dim),
        nn.BatchNorm1d(cfg.clustering_dim),
        nn.ReLU(),
        nn.Linear(cfg.clustering_dim, cfg.num_clusters)
    )

    model = model.apply(init_weights)

    model = model.to(cfg.device)

    return model

def init_weights(m):

    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.)

class FixedPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe) 

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x) 

class TransformerBatchNormEncoderLayer(nn.modules.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  
        src = src.permute(1, 2, 0)  
       
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  
        src = src.permute(1, 2, 0)  
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  
        return src
    
class SeqPooling(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.attention_pool = nn.Linear(embedding_dim, 1)
          
    def forward(self, x):
        w = self.attention_pool(x)
        w = F.softmax(w, dim=1)
        w = w.transpose(1, 2)
        
        y = torch.matmul(w, x)
        
        y = y.squeeze(1)
        
        return y

class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1, activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = FixedPositionalEncoding(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = F.gelu

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        
        self.seq_pooling = SeqPooling(self.d_model)
        
        self.projection_linear = nn.Linear(self.d_model, self.d_model // 2)
        self.projection_bn = nn.BatchNorm1d(self.d_model // 2)
        self.projection_relu = nn.ReLU(inplace=True)
        self.projection_fin_linear = nn.Linear(self.d_model // 2, self.d_model // 4)

    def forward(self, X, padding_masks, return_all = False):
        
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  
        inp = self.pos_enc(inp)  
      
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  
        output = self.act(output)  
        output_emb = output.permute(1, 0, 2)  
        output_pred = self.dropout1(output_emb)
        
        output_pred = self.output_layer(output_pred) 
        
        output_emb_pool = self.seq_pooling(output_emb)
        output_emb_proj = self.projection_linear(output_emb_pool)
        output_emb_fin_proj = self.projection_bn(output_emb_proj)
        output_emb_fin_proj = self.projection_relu(output_emb_fin_proj)
        output_emb_fin_proj = self.projection_fin_linear(output_emb_fin_proj)

        if return_all:
            return output_pred, output_emb, output_emb_pool, output_emb_proj, output_emb_fin_proj
            
        return output_pred, output_emb_fin_proj
    
class ClusteringModel(nn.Module):
    def __init__(self, nclusters):
        super(ClusteringModel, self).__init__()
        
        self.backbone_dim = 64
        self.intermediate_dim = 128
        self.output_dim = nclusters
        
        self.linear_first = nn.Linear(self.backbone_dim, self.intermediate_dim)
        self.activation = nn.ReLU()
        self.second_linear = nn.Linear(self.intermediate_dim, self.output_dim)
        self.bn = nn.BatchNorm1d(self.intermediate_dim)

    def forward(self, x):
        x = self.linear_first(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.second_linear(x)

        return x