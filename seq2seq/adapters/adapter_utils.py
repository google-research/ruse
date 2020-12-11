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

"""Implementation of different activation functions and hyper-network layers."""

import torch.nn as nn
from transformers.activations import get_activation


class Activations(nn.Module):
  def __init__(self, activation_type):
    super().__init__()
    self.f = get_activation(activation_type)

  def forward(self, x):
    return self.f(x)


def init_linear_layer(linear_layer, std=1e-2):
  """Initializes the given linear module as explained in adapter paper."""
  nn.init.normal_(linear_layer.weight, std=std)
  nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2):
  """Generates a linear module and initializes it."""
  linear = nn.Linear(input_dim, output_dim)
  init_linear_layer(linear, std=std)
  return linear


class HyperNetDownSampler(nn.Module):
  """This module generates the down sampler's weights for the meta adapter
  layers."""

  def __init__(self, config):
    super(HyperNetDownSampler, self).__init__()
    self.hidden_dim = config.hidden_dim
    self.input_dim = config.input_dim
    self.down_sample_size = self.input_dim // config.reduction_factor
    self.weight_generator = nn.Sequential(
      linear_layer(config.task_embedding_dim, self.hidden_dim),
      #nn.ReLU(),
      linear_layer(self.hidden_dim, self.input_dim * self.down_sample_size))
    self.bias_generator = nn.Sequential(
      linear_layer(config.task_embedding_dim, self.hidden_dim),
      #nn.ReLU(),
      linear_layer(self.hidden_dim, self.down_sample_size))

  def forward(self, task_embedding):
    task_embedding = task_embedding.view(-1)
    weight = self.weight_generator(task_embedding).view(self.down_sample_size, self.input_dim)
    bias = self.bias_generator(task_embedding).view(-1)
    return weight, bias


class HyperNetUpSampler(nn.Module):
  """This module generates the up sampler's weight for the meta adapter layers."""

  def __init__(self, config):
    super(HyperNetUpSampler, self).__init__()
    self.hidden_dim = config.hidden_dim
    self.input_dim = config.input_dim
    self.down_sample_size = self.input_dim // config.reduction_factor
    self.weight_generator = nn.Sequential(
      linear_layer(config.task_embedding_dim, self.hidden_dim),
      #nn.ReLU(),
      linear_layer(self.hidden_dim, self.input_dim * self.down_sample_size))
    self.bias_generator = nn.Sequential(
      linear_layer(config.task_embedding_dim, self.hidden_dim),
      #nn.ReLU(),
      linear_layer(self.hidden_dim, self.input_dim))

  def forward(self, task_embedding):
    task_embedding = task_embedding.view(-1)
    weight = self.weight_generator(task_embedding).view(self.input_dim, self.down_sample_size)
    bias = self.bias_generator(task_embedding).view(-1)
    return weight, bias
