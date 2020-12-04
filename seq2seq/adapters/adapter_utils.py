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

"""Implementation of different activation functions."""

import torch.nn as nn
from transformers.activations import get_activation

class Activations(nn.Module):
  def __init__(self, activation_type):
    super().__init__()
    self.f = get_activation(activation_type)

  def forward(self, x):
    return self.f(x)


class MetaDownSampler(nn.Module):
  def __init__(self, config):
    super(MetaDownSampler, self).__init__()
    self.input_dim = config.input_dim
    self.down_sample_size = config.down_sample_size
    self.weight_generator = nn.Sequential(
      nn.Linear(config.task_embedding_dim, config.hidden_dim),
      nn.ReLU(),
      nn.Linear(config.hidden_dim, self.input_dim))
    self.projection = nn.Linear(1, self.down_sample_size)
    # TODO: this can also be a MLP layer here.
    self.bias_generator = nn.Linear(config.task_embedding_dim, self.down_sample_size)

  def forward(self, task_embedding):
    z = self.weight_generator(task_embedding).reshape(-1, 1)
    weight = self.projection(z).transpose(0, 1)
    bias = self.bias_generator(task_embedding)
    return weight, bias


class MetaUpSampler(nn.Module):
  def __init__(self, config):
    super(MetaUpSampler, self).__init__()
    self.input_dim = config.input_dim
    self.down_sample_size = config.down_sample_size
    self.weight_generator = nn.Sequential(
      nn.Linear(config.task_embedding_dim, config.hidden_dim),
      nn.ReLU(),
      nn.Linear(config.hidden_dim, self.input_dim))
    self.projection = nn.Linear(1, self.down_sample_size)
    self.bias_generator = nn.Linear(config.task_embedding_dim, self.input_dim)

  def forward(self, task_embedding):
    z = self.weight_generator(task_embedding).reshape(-1, 1)
    weight = self.projection(z)
    bias = self.bias_generator(task_embedding)
    return weight, bias

