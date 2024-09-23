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
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, DIN_Attention, Dice, Sim_Score_Filter
import os
        

class DIN_db(BaseModel):
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
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_target_field=[("item_id", "cate_id")],
                 din_sequence_field=[("click_history", "cate_history")],
                 din_use_softmax=False,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DIN_db, self).__init__(feature_map,
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
        self.dnn_input_features = []
        for feature, feature_spec in feature_map.features.items():
            if feature_spec["source"] == "id":
                self.dnn_input_features.append(feature)
        assert len(self.din_target_field) == len(self.din_sequence_field), \
               "len(din_target_field) != len(din_sequence_field)"
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim, required_feature_columns=self.dnn_input_features)
        self.attention_layers = nn.ModuleList(
            [DIN_Attention(sum([self.feature_map.features[name].get("embedding_dim", self.embedding_dim) for name in target_field]) if type(target_field) == tuple \
                           else self.feature_map.features[target_field].get("embedding_dim", self.embedding_dim),  # == embedding_dim * 2
                           attention_units=attention_hidden_units,
                           hidden_activations=attention_hidden_activations,
                           output_activation=attention_output_activation,
                           dropout_rate=attention_dropout,
                           use_softmax=din_use_softmax)
             for target_field in self.din_target_field])
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(feature_source="id"),
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)

    def forward(self, inputs):
        X = self.get_inputs(inputs, feature_source="id")
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
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True, feature_source="id")  # concat and flatten all feature embeddings
        y_pred = self.dnn(feature_emb)
        return y_pred

    def get_embedding(self, field, feature_emb_dict):  # e.g. field == ('item_id', 'cate_id')
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)  # item and cate embeddings being concate
        else:
            return feature_emb_dict[field]


class Text_DB(BaseModel):
    def __init__(self, feature_map, **kwargs):
        super(Text_DB, self).__init__(feature_map, **kwargs)
        self.din = DIN_db(feature_map, **kwargs)
        self.din.load_weights(self.din.checkpoint)
        self.text_embedding_layer = FeatureEmbeddingDict(feature_map, kwargs["embedding_dim"], required_feature_columns=["item_emb", "item_emb_history"])
        self.filter = Sim_Score_Filter(feature_map,
                                       feature_sequence=False,
                                       **kwargs)
        self.alpha = kwargs["alpha"]
        self.compile(kwargs["optimizer"], kwargs["loss"], kwargs["learning_rate"])
        self.model_to_device()
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + "_all.model"))
        self.save_weights(self.checkpoint)

    def forward(self, inputs):
        y_pred_id = self.din(inputs)

        X_text = self.get_inputs(inputs, feature_source="text")
        feature_text_emb_dict = self.text_embedding_layer(X_text)

        sim_score = self.filter(X_text, feature_text_emb_dict)
        seq_len = torch.sum((sim_score != 0).detach(), dim=1)
        y_pred_text = torch.sum(sim_score * torch.arange(1, sim_score.shape[1] + 1).unsqueeze(0).to(self.device), dim=1) / seq_len
        y_pred_text = torch.sigmoid(y_pred_text).unsqueeze(-1)
        
        y_pred = {"y_pred": self.alpha * y_pred_id + (1 - self.alpha) * y_pred_text}
        return y_pred
