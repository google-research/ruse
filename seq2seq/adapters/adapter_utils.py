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

"""
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
"""


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
    """
    self.right_projection = nn.Sequential(
      nn.Linear(config.y_dim, self.hidden_dim),
      nn.ReLU(),
      nn.Linear(self.hidden_dim, self.down_sample_size)
    )
    """
    #self.projection = nn.Linear(1, self.down_sample_siz)
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
    #self.weight_layer_norm = nn.LayerNorm(self.input_dim, self.x_dim)
    #self.bias_layer_norm = nn.LayerNorm(self.down_sample_size)

    #nnn.Linear(config.task_embedding_dim, self.down_sample_size)

  def forward(self, task_embedding):
    #task_embedding = self.layernorm(task_embedding)
    task_embedding_reshaped = task_embedding.reshape(self.x_dim, self.y_dim)
    weight = self.left_projection(task_embedding_reshaped)
    #print("#### z ", z.shape)
    #weight = self.right_projection(z)
    bias = self.bias_generator(task_embedding).view(-1)
    #z = self.weight_generator(task_embedding).reshape(-1, 1)
    #weight = self.projection(z).transpose(0, 1)
    #bias = self.bias_generator(task_embedding)

    #weight = self.weight_layer_norm(weight)
    #bias = self.bias_layer_norm(bias)

    return weight, bias

class MetaUpSampler(nn.Module):
  def __init__(self, config):
    super(MetaUpSampler, self).__init__()
    self.hidden_dim = config.hidden_dim
    self.input_dim = config.input_dim
    self.x_dim = config.x_dim
    self.y_dim = config.y_dim
    #self.layernorm = nn.LayerNorm(config.task_embedding_dim)
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
    #self.weight_layer_norm = nn.LayerNorm(self.x_dim, self.input_dim)
    #self.bias_layer_norm = nn.LayerNorm(self.input_dim)

  def forward(self, task_embedding):
    #task_embedding = self.layernorm(task_embedding)
    task_embedding_reshaped = task_embedding.reshape(self.x_dim, self.y_dim)
    weight = self.left_projection(task_embedding_reshaped).transpose(0, 1)
    #print("#### z ", z.shape)
    #weight = self.right_projection(z)
    bias = self.bias_generator(task_embedding).view(-1)
    #z = self.weight_generator(task_embedding).reshape(-1, 1)
    #weight = self.projection(z).transpose(0, 1)
    #bias = self.bias_generator(task_embedding)

    #weight = self.weight_layer_norm(weight)
    #bias = self.bias_layer_norm(bias)

    return weight, bias

"""
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
"""

"""
if __name__ == "__main__":
  import torch
  from seq2seq.adapters import MetaAdapterConfig
  config = MetaAdapterConfig()
  config.input_dim = 512
  config.down_sample_size = 512//16
  down_sampler = MetaDownSampler(config)
  task_embedding = torch.zeros((1, 768))
  a, b = down_sampler(task_embedding)
  print(a.shape, b.shape)
  up_sampler = MetaUpSampler(config)
  a, b = up_sampler(task_embedding)
  print(a.shape)
  print(b.shape
  #a =  nn.Linear(config.task_embedding_dim, config.hidden_dim)
  #print(a.bias.shape)
"""

