import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .activations import Dice
from .blocks.mlp_block import MLP_Block

class DIN_Attention_score(nn.Module):
    def __init__(self, 
                 embedding_dim=64,
                 attention_units=[32], 
                 hidden_activations="ReLU",
                 output_activation=None,
                 dropout_rate=0,
                 batch_norm=False):
        super(DIN_Attention_score, self).__init__()
        self.embedding_dim = embedding_dim
        if isinstance(hidden_activations, str) and hidden_activations.lower() == "dice":
            hidden_activations = [Dice(units) for units in attention_units]
        self.attention_layer = MLP_Block(input_dim=4 * embedding_dim,
                                         output_dim=1,
                                         hidden_units=attention_units,
                                         hidden_activations=hidden_activations,
                                         output_activation=output_activation,
                                         dropout_rates=dropout_rate,
                                         batch_norm=batch_norm)

    def forward(self, target_item, history_sequence, mask=None):
        """
        target_item: b x emd
        history_sequence: b x len x emb
        mask: mask of history_sequence, 0 for masked positions
        """
        seq_len = history_sequence.size(1)
        target_item = target_item.unsqueeze(1).expand(-1, seq_len, -1)  # expand target_item embeddings to the same size as sequence
        attention_input = torch.cat([target_item, history_sequence, target_item - history_sequence, 
                                     target_item * history_sequence], dim=-1) # b x len x 4*emb
        attention_weight = self.attention_layer(attention_input.view(-1, 4 * self.embedding_dim))
        attention_weight = attention_weight.view(-1, seq_len) # b x len
        if mask is not None:
            attention_weight = attention_weight * mask.float()
        # if self.use_softmax:  # True
        #     if mask is not None:
        #         attention_weight += -1.e9 * (1 - mask.float())
        #     attention_weight = attention_weight.softmax(dim=-1)  # b x len, softmax in dim 'len'
        # output = (attention_weight.unsqueeze(-1) * history_sequence).sum(dim=1)  # sum pooling
        return attention_weight
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, Q, K, scale=None, mask=None):
        # mask: 0 for masked positions
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
        attention = scores.softmax(dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        return attention


class Attention_score(nn.Module):
    def __init__(self,
                 input_dim=64,
                 attention_dim=64,
                 num_heads=1,
                 dropout_rate=0,
                 use_scale=True):
        super(Attention_score, self).__init__()
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(dropout_rate)

    def forward(self, target_item, history_sequence, mask=None):
        """
        target_item: b x emd
        history_sequence: b x len x emb
        mask: mask of history_sequence, 0 for masked positions
        """
        # linear projection
        query = self.W_q(target_item)
        key = self.W_k(history_sequence)

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1)

        # scaled dot product attention
        attention_score = self.dot_attention(query, key, scale=self.scale, mask=mask)
        return attention_score


class Sim_Score_Filter(nn.Module):
    def __init__(self,
                 feature_map,
                 feature_sequence,
                 filter={},
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 embedding_dim=10,
                 t_attention_dim=64,
                 **kwargs):
        super(Sim_Score_Filter, self).__init__()
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.feature_sequence = feature_sequence
        self.filter_info = filter
        self.data_filter = False
        if self.filter_info["func"] == "cosine":
            self.similarity_function = F.cosine_similarity
        elif self.filter_info["func"] == "attn_score":
            self.similarity_function = DIN_Attention_score(embedding_dim=self.feature_map.features[self.filter_info["target_field"]].get("embedding_dim", self.embedding_dim),
                                                           attention_units=attention_hidden_units,
                                                           hidden_activations=attention_hidden_activations,
                                                           output_activation=attention_output_activation,
                                                           dropout_rate=attention_dropout)
        elif self.filter_info["func"] == "t_attn_score":
            self.similarity_function = Attention_score(input_dim=self.feature_map.features[self.filter_info["target_field"]].get("embedding_dim", self.embedding_dim),
                                                       attention_dim=t_attention_dim,
                                                       dropout_rate=attention_dropout)

        self.get_sim_score = kwargs.get("get_sim_score", False)
        if self.get_sim_score:
            self.sim_score_np = {"train": [], "eval": []}
        

    def forward(self, X, feature_emb_dict, epoch=0, stage="train"):
        if self.filter_info["func"] == "cosine":
            if self.filter_info["threshold"] != 0:
                target_expand = feature_emb_dict[self.filter_info["target_field"]].unsqueeze(1).expand(-1, feature_emb_dict[self.filter_info["sequence_field"]].size(1), -1)
                sim_score = self.similarity_function(target_expand, feature_emb_dict[self.filter_info["sequence_field"]], dim=2)
        elif self.filter_info["func"] in ["attn_score", "t_attn_score"]:
            sim_mask = X[self.filter_info["sequence_field"]].long() != 0
            sim_score = self.similarity_function(feature_emb_dict[self.filter_info["target_field"]],
                                                 feature_emb_dict[self.filter_info["sequence_field"]],
                                                 sim_mask)
        
        if self.data_filter or self.filter_info["filter_type"] == "sim_score_only":
            return sim_score
        
        sim_mask = self.get_sim_mask(sim_score)
        
        if self.filter_info.get("softmax", False):
            sim_score += -1.e9 * (1 - sim_mask.float())
            sim_score = sim_score.softmax(dim=-1)
        
        if self.filter_info["filter_type"] == "top_score": 
            self.multiply_score(feature_emb_dict, sim_score)
        self.multiply_score(feature_emb_dict, sim_mask)

        if epoch == 0 and self.get_sim_score:
            if self.sim_score_np:
                self.sim_score_np[stage].append(sim_score.detach().cpu().numpy())

        return feature_emb_dict, sim_mask
        
    def get_sim_mask(self, sim_score):
        if self.filter_info["threshold"] < 1:
            sim_mask = sim_score > self.filter_info["threshold"]
        else:
            sim_mask = torch.zeros_like(sim_score).bool()
            sim_score = sim_score.argsort(dim=1, descending=True)
            for i in range(sim_mask.shape[0]):
                sim_mask[i, sim_score[i, :self.filter_info["threshold"]]] = True
        return sim_mask
    
    def multiply_score(self, feature_emb_dict, score):
        for sequence_field in self.feature_sequence:
            if isinstance(sequence_field, tuple):
                for field in sequence_field:
                    feature_emb_dict[field] = feature_emb_dict[field] * score.unsqueeze(-1)
            else: feature_emb_dict[sequence_field] = feature_emb_dict[sequence_field] * score.unsqueeze(-1)

