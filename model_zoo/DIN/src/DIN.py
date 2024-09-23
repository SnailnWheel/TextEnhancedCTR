# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
import numpy as np
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, DIN_Attention, Dice


class DIN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DIN", 
                 gpu=-1, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_target_field=[("item_id", "cate_id")],
                 din_sequence_field=[("click_history", "cate_history")],
                 din_use_softmax=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DIN, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if not isinstance(din_target_field, list):
            din_target_field = [din_target_field]
        self.din_target_field = din_target_field
        if not isinstance(din_sequence_field, list):
            din_sequence_field = [din_sequence_field]
        self.din_sequence_field = din_sequence_field
        assert len(self.din_target_field) == len(self.din_sequence_field), \
               "len(din_target_field) != len(din_sequence_field)"
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.attention_layers = nn.ModuleList(
            [DIN_Attention(sum([self.feature_map.features[name].get("embedding_dim", self.embedding_dim) for name in target_field]) if type(target_field) == tuple \
                           else self.feature_map.features[target_field].get("embedding_dim", self.embedding_dim),  # == embedding_dim * 2
                           attention_units=attention_hidden_units,
                           hidden_activations=attention_hidden_activations,
                           output_activation=attention_output_activation,
                           dropout_rate=attention_dropout,
                           use_softmax=din_use_softmax)
             for target_field in self.din_target_field])
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field, 
                                                                 self.din_sequence_field)):
            target_emb = self.get_embedding(target_field, feature_emb_dict)  # here embedding_dim = embedding_dim * 2
            sequence_emb = self.get_embedding(sequence_field, feature_emb_dict)
            seq_field = list(flatten([sequence_field]))[0] # flatten nested list to pick the first sequence field: 'item_history'
            mask = X[seq_field].long() != 0 # padding_idx = 0 required, here 0 means no data
            pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask)
            for field, field_emb in zip(list(flatten([sequence_field])),
                                        pooling_emb.split([self.feature_map.features[f].get("embedding_dim", self.embedding_dim) for f in list(flatten([sequence_field]))], dim=-1)):
                feature_emb_dict[field] = field_emb
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)  # concat and flatten all feature embeddings
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_embedding(self, field, feature_emb_dict):  # e.g. field == ('item_id', 'cate_id')
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)  # item and cate embeddings being concate
        else:
            return feature_emb_dict[field]


class DIN_wadd(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DIN_wadd", 
                 gpu=-1, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_target_field=[("item_id", "cate_id")],
                 din_sequence_field=[("click_history", "cate_history")],
                 din_use_softmax=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DIN_wadd, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if not isinstance(din_target_field, list):
            din_target_field = [din_target_field]
        self.din_target_field = din_target_field
        if not isinstance(din_sequence_field, list):
            din_sequence_field = [din_sequence_field]
        self.din_sequence_field = din_sequence_field
        assert len(self.din_target_field) == len(self.din_sequence_field), \
               "len(din_target_field) != len(din_sequence_field)"
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        mlp_fusion_id_hidden_units = kwargs["mlp_fusion_id_hidden_units"] + [embedding_dim]
        if isinstance(kwargs["mlp_fusion_id_activations"], str) and kwargs["mlp_fusion_id_activations"].lower() == "dice":
            mlp_fusion_id_activations = [Dice(units) for units in mlp_fusion_id_hidden_units]
        self.mlp_fusion_id_list = kwargs["mlp_fusion_id"]
        self.weighted_add = kwargs["weighted_add"]
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.mlp_fusion_id = MLP_Block(input_dim=embedding_dim + embedding_dim,
                                   hidden_units=mlp_fusion_id_hidden_units,
                                   hidden_activations=mlp_fusion_id_activations,
                                   dropout_rates=net_dropout,
                                   batch_norm=batch_norm)
        self.add_weight_1 = nn.Parameter(torch.ones(1))
        self.add_weight_2 = nn.Parameter(torch.ones(1))
        self.attention_layers = nn.ModuleList(
            [DIN_Attention(sum([self.feature_map.features[name].get("embedding_dim", self.embedding_dim) for name in target_field]) if type(target_field) == tuple \
                           else self.feature_map.features[target_field].get("embedding_dim", self.embedding_dim),  # == embedding_dim * 2
                           attention_units=attention_hidden_units,
                           hidden_activations=attention_hidden_activations,
                           output_activation=attention_output_activation,
                           dropout_rate=attention_dropout,
                           use_softmax=din_use_softmax)
             for target_field in self.din_target_field])
        self.dnn = MLP_Block(input_dim=self.embedding_dim * 4,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        # mlp_fusion_id
        self.fusion_module(feature_emb_dict, self.mlp_fusion_id_list)
        # weighted_add
        for add_list in self.weighted_add:
            add_target_name = add_list[0]
            feature_emb_dict[add_target_name] = self.add_weight_1 * feature_emb_dict[add_list[0]] + self.add_weight_2 * feature_emb_dict[add_list[1]]
        # din
        for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field, 
                                                                 self.din_sequence_field)):
            target_emb = self.get_embedding(target_field, feature_emb_dict)  # here embedding_dim = embedding_dim * 2
            sequence_emb = self.get_embedding(sequence_field, feature_emb_dict)
            seq_field = list(flatten([sequence_field]))[0] # flatten nested list to pick the first sequence field: 'item_history'
            mask = X[seq_field].long() != 0 # padding_idx = 0 required, here 0 means no data
            pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask)
            for field, field_emb in zip(list(flatten([sequence_field])),
                                        pooling_emb.split([self.feature_map.features[f].get("embedding_dim", self.embedding_dim) for f in list(flatten([sequence_field]))], dim=-1)):
                feature_emb_dict[field] = field_emb
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True, feature_list=["user_id", "user_emb", "item_emb", "item_emb_history"])  # concat and flatten all feature embeddings
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_embedding(self, field, feature_emb_dict):  # e.g. field == ('item_id', 'cate_id')
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)  # item and cate embeddings being concate
        else:
            return feature_emb_dict[field]
        
    def fusion_module(self, feature_emb_dict, id_lists):
        for id_list in id_lists:
            target_name = id_list[0]
            is_sequence = self.feature_map.features[target_name]["type"] == "sequence"
            mlp_fusion_id_input = torch.cat([feature_emb_dict[name].view(-1, self.embedding_dim) for name in id_list], dim=-1) if is_sequence \
                else torch.cat([feature_emb_dict[name] for name in id_list], dim=-1)
            mlp_fusion_id_output = self.mlp_fusion_id(mlp_fusion_id_input)
            feature_emb_dict[target_name] = mlp_fusion_id_output.view(-1, self.feature_map.features[target_name]["max_len"], self.embedding_dim) if is_sequence \
                else mlp_fusion_id_output


class DIN_MLP_fusion(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DIN_wadd", 
                 gpu=-1, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_target_field=[("item_id", "cate_id")],
                 din_sequence_field=[("click_history", "cate_history")],
                 din_use_softmax=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DIN_MLP_fusion, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if not isinstance(din_target_field, list):
            din_target_field = [din_target_field]
        self.din_target_field = din_target_field
        if not isinstance(din_sequence_field, list):
            din_sequence_field = [din_sequence_field]
        self.din_sequence_field = din_sequence_field
        assert len(self.din_target_field) == len(self.din_sequence_field), \
               "len(din_target_field) != len(din_sequence_field)"
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        
        self.mlp_fusion_id_output_dim = kwargs["mlp_fusion_id_output_dim"]
        mlp_fusion_id_hidden_units = kwargs["mlp_fusion_id_hidden_units"] + [self.mlp_fusion_id_output_dim]
        if isinstance(kwargs["mlp_fusion_id_activations"], str) and kwargs["mlp_fusion_id_activations"].lower() == "dice":
            mlp_fusion_id_activations = [Dice(units) for units in mlp_fusion_id_hidden_units]
        
        self.mlp_fusion_output_dim = kwargs["mlp_fusion_output_dim"]
        mlp_fusion_hidden_units = kwargs["mlp_fusion_hidden_units"] + [self.mlp_fusion_output_dim]
        if isinstance(kwargs["mlp_fusion_activations"], str) and kwargs["mlp_fusion_activations"].lower() == "dice":
            mlp_fusion_activations = [Dice(units) for units in mlp_fusion_hidden_units]
        
        self.mlp_fusion_id_list = kwargs["mlp_fusion_id"]
        self.mlp_fusion_list = kwargs["mlp_fusion"]
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.mlp_fusion_id = MLP_Block(input_dim=self.embedding_dim + self.embedding_dim,
                                   hidden_units=mlp_fusion_id_hidden_units,
                                   hidden_activations=mlp_fusion_id_activations,
                                   dropout_rates=net_dropout,
                                   batch_norm=batch_norm)
        self.mlp_fusion = MLP_Block(input_dim=sum([self.feature_map.features[name].get("embedding_dim", self.embedding_dim) for name in self.mlp_fusion_list[0]]),
                                   hidden_units=mlp_fusion_hidden_units,
                                   hidden_activations=mlp_fusion_activations,
                                   dropout_rates=net_dropout,
                                   batch_norm=batch_norm)
        self.attention_layers = nn.ModuleList(
            [DIN_Attention(sum([self.feature_map.features[name].get("embedding_dim", self.embedding_dim) for name in target_field]) if type(target_field) == tuple \
                           else self.mlp_fusion_output_dim,  # == embedding_dim * 2
                           attention_units=attention_hidden_units,
                           hidden_activations=attention_hidden_activations,
                           output_activation=attention_output_activation,
                           dropout_rate=attention_dropout,
                           use_softmax=din_use_softmax)
             for target_field in self.din_target_field])
        self.dnn = MLP_Block(input_dim=self.feature_map.features["user_id"].get("embedding_dim", self.embedding_dim) \
                                + self.feature_map.features["user_emb"].get("embedding_dim", self.embedding_dim) \
                                + self.mlp_fusion_output_dim * 2,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        # mlp_fusion_id
        self.fusion_module(feature_emb_dict, self.mlp_fusion_id_list, self.mlp_fusion_id)
        # mlp_fusion
        self.fusion_module(feature_emb_dict, self.mlp_fusion_list, self.mlp_fusion)
        # din
        for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field, 
                                                                 self.din_sequence_field)):
            target_emb = self.get_embedding(target_field, feature_emb_dict)  # here embedding_dim = embedding_dim * 2
            sequence_emb = self.get_embedding(sequence_field, feature_emb_dict)
            seq_field = list(flatten([sequence_field]))[0] # flatten nested list to pick the first sequence field: 'item_history'
            mask = X[seq_field].long() != 0 # padding_idx = 0 required, here 0 means no data
            pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask)
            feature_emb_dict[sequence_field] = pooling_emb
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True, feature_list=["user_id", "user_emb", "item_emb", "item_emb_history"])  # concat and flatten all feature embeddings
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_embedding(self, field, feature_emb_dict):  # e.g. field == ('item_id', 'cate_id')
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)  # item and cate embeddings being concate
        else:
            return feature_emb_dict[field]
        
    def fusion_module(self, feature_emb_dict, id_lists, model):
        for id_list in id_lists:
            target_name = id_list[0]
            is_sequence = self.feature_map.features[target_name]["type"] == "sequence"
            mlp_fusion_id_input = torch.cat([feature_emb_dict[name].view(-1, self.feature_map.features[name].get("embedding_dim", self.embedding_dim)) for name in id_list], dim=-1) if is_sequence \
                else torch.cat([feature_emb_dict[name] for name in id_list], dim=-1)
            mlp_fusion_id_output = model(mlp_fusion_id_input)
            feature_emb_dict[target_name] = mlp_fusion_id_output.view(-1, self.feature_map.features[target_name]["max_len"], self.embedding_dim) if is_sequence \
                else mlp_fusion_id_output

            
class DIN_MLP_fusion_b(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DIN_wadd", 
                 gpu=-1, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_target_field=[("item_id", "cate_id")],
                 din_sequence_field=[("click_history", "cate_history")],
                 din_use_softmax=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DIN_MLP_fusion_b, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if not isinstance(din_target_field, list):
            din_target_field = [din_target_field]
        self.din_target_field = din_target_field
        if not isinstance(din_sequence_field, list):
            din_sequence_field = [din_sequence_field]
        self.din_sequence_field = din_sequence_field
        assert len(self.din_target_field) == len(self.din_sequence_field), \
               "len(din_target_field) != len(din_sequence_field)"
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        self.mlp_fusion_output_dim = kwargs["mlp_fusion_output_dim"]
        mlp_fusion_hidden_units = kwargs["mlp_fusion_hidden_units"] + [self.mlp_fusion_output_dim]
        if isinstance(kwargs["mlp_fusion_activations"], str) and kwargs["mlp_fusion_activations"].lower() == "dice":
            mlp_fusion_activations = [Dice(units) for units in mlp_fusion_hidden_units]
        self.mlp_fusion_list = kwargs["mlp_fusion"]
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.mlp_fusion = MLP_Block(input_dim=sum([self.feature_map.features[name].get("embedding_dim", self.embedding_dim) for name in self.mlp_fusion_list[0]]),
                                   hidden_units=mlp_fusion_hidden_units,
                                   hidden_activations=mlp_fusion_activations,
                                   dropout_rates=net_dropout,
                                   batch_norm=batch_norm)
        self.attention_layers = nn.ModuleList(
            [DIN_Attention(sum([self.feature_map.features[name].get("embedding_dim", self.embedding_dim) for name in target_field]) if type(target_field) == tuple \
                           else self.mlp_fusion_output_dim,  # == embedding_dim * 2
                           attention_units=attention_hidden_units,
                           hidden_activations=attention_hidden_activations,
                           output_activation=attention_output_activation,
                           dropout_rate=attention_dropout,
                           use_softmax=din_use_softmax)
             for target_field in self.din_target_field])
        self.dnn = MLP_Block(input_dim=self.feature_map.features["user_id"].get("embedding_dim", self.embedding_dim) \
                                + self.feature_map.features["user_emb"].get("embedding_dim", self.embedding_dim) \
                                + self.mlp_fusion_output_dim * 2,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        # mlp_fusion
        self.fusion_module(feature_emb_dict, self.mlp_fusion_list, self.mlp_fusion)
        # din
        for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field, 
                                                                 self.din_sequence_field)):
            target_emb = self.get_embedding(target_field, feature_emb_dict)  # here embedding_dim = embedding_dim * 2
            sequence_emb = self.get_embedding(sequence_field, feature_emb_dict)
            seq_field = list(flatten([sequence_field]))[0] # flatten nested list to pick the first sequence field: 'item_history'
            mask = X[seq_field].long() != 0 # padding_idx = 0 required, here 0 means no data
            pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask)
            feature_emb_dict[sequence_field] = pooling_emb
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True, feature_list=["user_id", "user_emb", "item_emb", "item_emb_history"])  # concat and flatten all feature embeddings
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_embedding(self, field, feature_emb_dict):  # e.g. field == ('item_id', 'cate_id')
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)  # item and cate embeddings being concate
        else:
            return feature_emb_dict[field]
        
    def fusion_module(self, feature_emb_dict, id_lists, model):
        for id_list in id_lists:
            target_name = id_list[0]
            is_sequence = self.feature_map.features[target_name]["type"] == "sequence"
            mlp_fusion_id_input = torch.cat([feature_emb_dict[name].view(-1, self.feature_map.features[name].get("embedding_dim", self.embedding_dim)) for name in id_list], dim=-1) if is_sequence \
                else torch.cat([feature_emb_dict[name] for name in id_list], dim=-1)
            mlp_fusion_id_output = model(mlp_fusion_id_input)
            feature_emb_dict[target_name] = mlp_fusion_id_output.view(-1, self.feature_map.features[target_name]["max_len"], self.embedding_dim) if is_sequence \
                else mlp_fusion_id_output
            

class DIN_MLP_adapter(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DIN_wadd", 
                 gpu=-1, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_target_field=[("item_id", "cate_id")],
                 din_sequence_field=[("click_history", "cate_history")],
                 din_use_softmax=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DIN_MLP_adapter, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if not isinstance(din_target_field, list):
            din_target_field = [din_target_field]
        self.din_target_field = din_target_field
        if not isinstance(din_sequence_field, list):
            din_sequence_field = [din_sequence_field]
        self.din_sequence_field = din_sequence_field
        assert len(self.din_target_field) == len(self.din_sequence_field), \
               "len(din_target_field) != len(din_sequence_field)"
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]

        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)

        self.id_adapter_list = kwargs["id_adapter"]
        id_adapter_hidden_units = kwargs["id_adapter_hidden_units"]
        if isinstance(kwargs["id_adapter_activations"], str) and kwargs["id_adapter_activations"].lower() == "dice":
            id_adapter_activations = [Dice(units) for units in id_adapter_hidden_units]
        self.id_adapter = MLP_Block(input_dim=sum([self.feature_map.features[name].get("embedding_dim", self.embedding_dim) for name in self.id_adapter_list[0]]),
                                   hidden_units=id_adapter_hidden_units,
                                   hidden_activations=id_adapter_activations,
                                   dropout_rates=net_dropout,
                                   batch_norm=batch_norm)

        self.text_adapter_list = kwargs["text_adapter"]
        text_adapter_hidden_units = kwargs["text_adapter_hidden_units"]
        if isinstance(kwargs["text_adapter_activations"], str) and kwargs["text_adapter_activations"].lower() == "dice":
            text_adapter_activations = [Dice(units) for units in text_adapter_hidden_units]
        self.text_adapter = MLP_Block(input_dim=sum([self.feature_map.features[name].get("embedding_dim", self.embedding_dim) for name in self.text_adapter_list[0]]),
                                   hidden_units=text_adapter_hidden_units,
                                   hidden_activations=text_adapter_activations,
                                   dropout_rates=net_dropout,
                                   batch_norm=batch_norm)

        self.attn_input_dim = {
            self.id_adapter_list[0][0]  : id_adapter_hidden_units[-1],
            self.text_adapter_list[0][0]: text_adapter_hidden_units[-1]
            }
        self.attention_layers = nn.ModuleList(
            [DIN_Attention(sum(self.attn_input_dim.values()) if type(target_field) == tuple \
                           else self.attn_input_dim[target_field],  # == embedding_dim * 2
                           attention_units=attention_hidden_units,
                           hidden_activations=attention_hidden_activations,
                           output_activation=attention_output_activation,
                           dropout_rate=attention_dropout,
                           use_softmax=din_use_softmax)
             for target_field in self.din_target_field])
        self.dnn = MLP_Block(input_dim=self.feature_map.features["user_id"].get("embedding_dim", self.embedding_dim) \
                                     + self.feature_map.features["user_emb"].get("embedding_dim", self.embedding_dim) \
                                     + sum(self.attn_input_dim.values()) * 2,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        # adapter
        self.fusion_module(feature_emb_dict, id_lists=self.id_adapter_list, model=self.id_adapter)
        self.fusion_module(feature_emb_dict, id_lists=self.text_adapter_list, model=self.text_adapter)
        # din
        for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field, 
                                                                 self.din_sequence_field)):
            target_emb = self.get_embedding(target_field, feature_emb_dict)  # here embedding_dim = embedding_dim * 2
            sequence_emb = self.get_embedding(sequence_field, feature_emb_dict)
            seq_field = list(flatten([sequence_field]))[0] # flatten nested list to pick the first sequence field: 'item_history'
            mask = X[seq_field].long() != 0 # padding_idx = 0 required, here 0 means no data
            pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask)
            for field, field_emb in zip(list(flatten([sequence_field])),
                                        pooling_emb.split([self.attn_input_dim[f] for f in list(flatten([target_field]))], dim=-1)):
                feature_emb_dict[field] = field_emb
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True, feature_list=["user_id", "user_emb", "item_id", "item_history", "item_emb", "item_emb_history"])  # concat and flatten all feature embeddings
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_embedding(self, field, feature_emb_dict):  # e.g. field == ('item_id', 'cate_id')
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)  # item and cate embeddings being concate
        else:
            return feature_emb_dict[field]
        
    def fusion_module(self, feature_emb_dict, id_lists, model):
        output_dim = self.attn_input_dim[id_lists[0][0]]
        for id_list in id_lists:
            target_name = id_list[0]
            is_sequence = self.feature_map.features[target_name]["type"] == "sequence"
            mlp_fusion_id_input = torch.cat([feature_emb_dict[name].view(-1, self.feature_map.features[name].get("embedding_dim", self.embedding_dim)) for name in id_list], dim=-1) if is_sequence \
                else torch.cat([feature_emb_dict[name] for name in id_list], dim=-1)
            mlp_fusion_id_output = model(mlp_fusion_id_input)
            feature_emb_dict[target_name] = mlp_fusion_id_output.view(-1, self.feature_map.features[target_name]["max_len"], output_dim) if is_sequence \
                else mlp_fusion_id_output
