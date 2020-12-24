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
import torch
import numpy as np
import os

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
    self.task_embedding_dim = config.projected_task_embedding_dim if config.train_task_embeddings else config.task_embedding_dim
    self.weight_generator = nn.Sequential(
      linear_layer(self.task_embedding_dim, self.hidden_dim),
      linear_layer(self.hidden_dim, self.input_dim * self.down_sample_size))
    self.bias_generator = nn.Sequential(
      linear_layer(self.task_embedding_dim, self.hidden_dim),
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
    self.train_task_embeddings = config.train_task_embeddings
    self.task_embedding_dim = config.projected_task_embedding_dim if config.train_task_embeddings else config.task_embedding_dim
    self.weight_generator = nn.Sequential(
      linear_layer(self.task_embedding_dim, self.hidden_dim),
      linear_layer(self.hidden_dim, self.input_dim * self.down_sample_size))
    self.bias_generator = nn.Sequential(
      linear_layer(self.task_embedding_dim, self.hidden_dim),
      linear_layer(self.hidden_dim, self.input_dim))

  def forward(self, task_embedding):
    task_embedding = task_embedding.view(-1)
    weight = self.weight_generator(task_embedding).view(self.input_dim, self.down_sample_size)
    bias = self.bias_generator(task_embedding).view(-1)
    return weight, bias


class TaskHyperNet(nn.Module):
  """This module generates the task-embeddings from the initial feeded task embeddings."""

  def __init__(self, config):
    super(TaskHyperNet, self).__init__()
    self.task_hidden_dim = config.task_hidden_dim
    self.projected_task_embedding_dim = config.projected_task_embedding_dim
    self.task_embeding_generator = nn.Sequential(
      linear_layer(config.task_embedding_dim, self.task_hidden_dim),
      #nn.ReLU(),
      linear_layer(self.task_hidden_dim, self.projected_task_embedding_dim))

  def forward(self, task_embedding):
    task_embedding = task_embedding.view(-1)
    return self.task_embeding_generator(task_embedding).view(-1)


class LayerNormHyperNet(nn.Module):
  """This module generates the weight and bias for the task conditioned layer norm."""
  def __init__(self, config):
    super(LayerNormHyperNet, self).__init__()
    self.task_embedding_dim = config.projected_task_embedding_dim if config.train_task_embeddings else config.task_embedding_dim
    self.weight_generator = linear_layer(self.task_embedding_dim, config.input_dim)
    self.bias_generator = linear_layer(self.task_embedding_dim, config.input_dim)

  def forward(self, input):
    return self.weight_generator(input), self.bias_generator(input)


class TaskEmbeddingController(nn.Module):
  def __init__(self, config):
    super(TaskEmbeddingController, self).__init__()
    self.device = config.device 
    self.task_embedding_dim = config.task_embedding_dim
    self.task_embedding_dir = config.task_embedding_dir
    self.set_task_embeddings(config.tasks)
    self.train_task_embeddings = config.train_task_embeddings
    if self.train_task_embeddings:
      self.task_hyper_net = TaskHyperNet(config)

  # Defines utilities for task-embeddings.
  def load_or_init_task_embedding(self, task):
      if self.task_embedding_dir is not None:
        task_embedding_path = os.path.join(self.task_embedding_dir, task + ".npy")
        return torch.Tensor(np.load(task_embedding_path)).to(self.device)
      else:
        return torch.Tensor(torch.randn(self.task_embedding_dim)).to(self.device)

  def set_task_embeddings(self, tasks, parametric=False):
      self.task_to_embeddings = {} if not parametric else nn.ParameterDict(dict())
      for task in tasks:
        task_embedding = self.load_or_init_task_embedding(task)
        self.task_to_embeddings[task] = task_embedding if not parametric else nn.Parameter(task_embedding)

  def update_task_embeddings(self, tasks, parametric=False):
      self.task_to_embeddings = {} if not parametric else nn.ParameterDict(dict())
      for task in tasks:
        # if task not in self.task_to_embeddings:
        task_embedding = self.load_or_init_task_embedding(task)
        self.task_to_embeddings[task] = task_embedding if not parametric else nn.Parameter(task_embedding)

  def forward(self, task):
      task_embedding = self.task_to_embeddings[task]
      if self.train_task_embeddings:
        return self.task_hyper_net(task_embedding)
      return task_embedding


