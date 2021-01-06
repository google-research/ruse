# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines pooling layers for pooling encoder hidden states to
learn fixed length sentence representations."""

import abc
import math
from collections import OrderedDict

import torch
from torch import nn


class Pooling(nn.Module, metaclass=abc.ABCMeta):
    """Pooling Layer Abstract Base class."""

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(self):
        """This method implements pooling of the hidden_states of shape
        (batch_size x seq_length x hidden_size) and returns a fixed length
         sentence representation of shape (batch_size x hidden_size x 1)"""
        raise NotImplementedError()


class MeanPooling(Pooling):
    """Applies mean pooling on the hidden_states."""

    def forward(self, hidden_states, attention_mask):
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden_states = torch.sum(hidden_states * attention_mask_expanded, 1)
        sum_attention_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        return (sum_hidden_states / sum_attention_mask).unsqueeze(-1)


class MaxPooling(Pooling):
    """Applied max pooling on the hidden_states."""

    def forward(self, hidden_states, attention_mask):
        return hidden_states.max(dim=1, keepdim=True)[0].transpose(1, 2)


class AttentivePooling(Pooling):
    """Applies attention pooling on the hidden_states."""

    def __init__(self, config):
        super(AttentivePooling, self).__init__(config)
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                "The d_model size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.d_model, config.num_heads))
        self.num_heads = config.num_heads
        self.attention_head_size = int(config.d_model / config.num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size
        self.query = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.key = nn.Linear(config.d_model, self.all_head_size)
        self.value = nn.Linear(config.d_model, self.all_head_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        batch_size = hidden_states.shape[0]
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        mixed_query_layer = self.query.repeat(batch_size, 1, 1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention
        # scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        outputs = context_layer.view(*new_context_layer_shape)
        return outputs.transpose(1, 2)


POOLING_MAPPING = OrderedDict([("mean", MeanPooling),
                               ("max", MaxPooling),
                               ("attentive", AttentivePooling)])


class AutoPooling(nn.Module):
    """Generic pooling class to instantiate different pooling classes."""

    @classmethod
    def get(cls, pooling_type: str, config, *args, **kwargs):
        if pooling_type in POOLING_MAPPING:
            pooling_class = POOLING_MAPPING[pooling_type]
            return pooling_class(config, *args, **kwargs)
        raise ValueError(
            "Unrecognized pooling type identifier: {}. Should contain one of {}"
                .format(pooling_type, ", ".join(POOLING_MAPPING.keys())))
