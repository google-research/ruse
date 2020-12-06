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



def init_linear_layer(linear_layer, std=1e-2):
  """Initializes the linear modules as explained in adapter paper."""
  nn.init.normal_(linear_layer.weight, std=std)
  nn.init.zeros_(linear_layer.bias)


class MetaDownSampler(nn.Module):
  def __init__(self, config):
    super(MetaDownSampler, self).__init__()
    self.hidden_dim = config.hidden_dim
    self.input_dim = config.input_dim
    self.x_dim = config.x_dim
    self.y_dim = config.y_dim
    self.down_sample_size = config.down_sample_size
    linear1 = nn.Linear(config.y_dim, config.hidden_dim)
    init_linear_layer(linear1)
    linear2 = nn.Linear(config.hidden_dim, self.input_dim)
    init_linear_layer(linear2)
    self.left_projection = nn.Sequential(
      linear1,
      nn.ReLU(),
      linear2)
    # TODO: this can also be a MLP layer here.
    linear3 = nn.Linear(config.task_embedding_dim, self.hidden_dim)
    init_linear_layer(linear3)
    linear4 = nn.Linear(self.hidden_dim, self.down_sample_size)
    init_linear_layer(linear4)
    self.bias_generator = nn.Sequential(
      linear3,
      nn.ReLU(),
      linear4
    )

  def forward(self, task_embedding):
    task_embedding_reshaped = task_embedding.reshape(self.x_dim, self.y_dim)
    weight = self.left_projection(task_embedding_reshaped)
    bias = self.bias_generator(task_embedding).view(-1)
    return weight, bias


class MetaUpSampler(nn.Module):
  def __init__(self, config):
    super(MetaUpSampler, self).__init__()
    self.hidden_dim = config.hidden_dim
    self.input_dim = config.input_dim
    self.x_dim = config.x_dim
    self.y_dim = config.y_dim
    self.down_sample_size = config.down_sample_size
    linear1 = nn.Linear(config.y_dim, config.hidden_dim)
    init_linear_layer(linear1)
    linear2 = nn.Linear(config.hidden_dim, self.input_dim)
    init_linear_layer(linear2)
    self.left_projection = nn.Sequential(
      linear1,
      nn.ReLU(),
      linear2)
    linear3 = nn.Linear(config.task_embedding_dim, self.hidden_dim)
    init_linear_layer(linear3)
    linear4 = nn.Linear(self.hidden_dim, self.input_dim)
    init_linear_layer(linear4)
    self.bias_generator = nn.Sequential(
      linear3,
      nn.ReLU(),
      linear4
    )

  def forward(self, task_embedding):
    task_embedding_reshaped = task_embedding.reshape(self.x_dim, self.y_dim)
    weight = self.left_projection(task_embedding_reshaped).transpose(0, 1)
    bias = self.bias_generator(task_embedding).view(-1)
    return weight, bias


class MetaParameterizedDownSampler(nn.Module):
  def __init__(self, config):
    super(MetaParameterizedDownSampler, self).__init__()
    self.hidden_dim = config.hidden_dim
    self.input_dim = config.input_dim
    self.x_dim = config.x_dim
    self.y_dim = config.y_dim
    reduction_factor = config.reduction_factor if config.reduction_factor is not None else 2
    self.down_sample_size = self.input_dim // reduction_factor
    linear1 = nn.Linear(config.x_dim, config.hidden_dim)
    init_linear_layer(linear1)
    linear2 = nn.Linear(config.hidden_dim, self.input_dim * self.down_sample_size // config.y_dim)
    init_linear_layer(linear2)
    self.left_projection = nn.Sequential(
      linear1,
      nn.ReLU(),
      linear2)
    linear3 = nn.Linear(config.task_embedding_dim, self.hidden_dim)
    init_linear_layer(linear3)
    linear4 = nn.Linear(self.hidden_dim, self.down_sample_size)
    init_linear_layer(linear4)
    self.bias_generator = nn.Sequential(
      linear3,
      nn.ReLU(),
      linear4
    )

  def forward(self, task_embedding):
    task_embedding_reshaped = task_embedding.reshape(self.x_dim, self.y_dim)
    weight = self.left_projection(task_embedding_reshaped).view(self.down_sample_size, self.input_dim)
    task_embedding = task_embedding.view(-1)
    bias = self.bias_generator(task_embedding).view(-1)
    return weight, bias


class MetaParameterizedUpSampler(nn.Module):
  def __init__(self, config):
    super(MetaParameterizedUpSampler, self).__init__()
    self.hidden_dim = config.hidden_dim
    self.input_dim = config.input_dim
    self.x_dim = config.x_dim
    self.y_dim = config.y_dim
    reduction_factor = config.reduction_factor if config.reduction_factor is not None else 2
    self.down_sample_size = self.input_dim // reduction_factor
    linear1 = nn.Linear(config.y_dim, config.hidden_dim)
    init_linear_layer(linear1)
    linear2 = nn.Linear(config.hidden_dim, self.input_dim * self.down_sample_size // config.y_dim)
    init_linear_layer(linear2)
    self.left_projection = nn.Sequential(
      linear1,
      nn.ReLU(),
      linear2)
    linear3 = nn.Linear(config.task_embedding_dim, self.hidden_dim)
    init_linear_layer(linear3)
    linear4 = nn.Linear(self.hidden_dim, self.input_dim)
    init_linear_layer(linear4)
    self.bias_generator = nn.Sequential(
      linear3,
      nn.ReLU(),
      linear4)

  def forward(self, task_embedding):
    task_embedding_reshaped = task_embedding.reshape(self.x_dim, self.y_dim)
    weight = self.left_projection(task_embedding_reshaped).view(self.input_dim, self.down_sample_size)
    task_embedding = task_embedding.view(-1)
    bias = self.bias_generator(task_embedding).view(-1)
    return weight, bias
