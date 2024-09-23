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
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, CrossNet, MaskedAveragePooling, Sim_Score_Filter


class DCN_filter(BaseModel):
    def __init__(self, 
                 feature_map,
                 model_id="DCN",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DCN_filter, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)

        self.feature_encoders = nn.ModuleDict()
        self.feature_sequence = []
        for feature, feature_spec in feature_map.features.items():
            if feature_spec["source"] == "id" and feature_spec["type"] == "sequence":
                self.feature_encoders[feature] = MaskedAveragePooling()
                self.feature_sequence.append(feature)

        self.filter = Sim_Score_Filter(feature_map,
                                       self.feature_sequence,
                                       embedding_dim=embedding_dim,
                                       **kwargs)

        input_dim = feature_map.sum_emb_out_dim(feature_source="id")
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case of only crossing net used
        self.crossnet = CrossNet(input_dim, num_cross_layers)
        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0: # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.fc = nn.Linear(final_dim, 1) # [cross_part, dnn_part] -> logit
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)

        # calculate similarity score
        feature_emb_dict, _ = self.filter(X, feature_emb_dict, self._epoch_index, self.stage)
        
        for feature in self.feature_sequence:
            feature_emb_dict[feature] = self.feature_encoders[feature](feature_emb_dict[feature])
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True, feature_source="id")

        cross_out = self.crossnet(feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        y_pred_1 = self.fc(final_out)
        y_pred = self.output_activation(y_pred_1)
        # ---------------------------
        if not ((y_pred >= 0).all() and (y_pred <= 1).all()):
            y_pred_np = y_pred_1.detach().cpu().numpy()
            np.savetxt("./y_pred_1_np.csv", y_pred_np, delimiter=',')
            print(y_pred_np)
            print(self.output_activation)
            y_sig_np = 1 / (1 + np.exp(- y_pred_np))
            np.savetxt("./y_sig_np.csv", y_sig_np, delimiter=',')
            print(y_sig_np)
            print((y_sig_np >= 0).all() and (y_sig_np <= 1).all())
            print(y_pred.detach().cpu().numpy())
            np.savetxt("./y_sig.csv", y_pred.detach().cpu().numpy(), delimiter=',')
        #----------------------------
        assert (y_pred >= 0).all() and (y_pred <= 1).all(), "Output out of range"
        return_dict = {"y_pred": y_pred}
        return return_dict

