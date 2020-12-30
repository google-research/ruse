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


import numpy as np
import os
import torch
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


class AdapterHyperNet(nn.Module):
    """This module generates the weights for the meta adapter layers."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterHyperNet, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_task_embeddings = config.train_task_embeddings
        self.task_embedding_dim = config.projected_task_embedding_dim if config.train_task_embeddings else config.task_embedding_dim
        self.one_layer_adapter_hyper_net = config.one_layer_adapter_hyper_net
        self.adapter_hyper_net_with_bias = config.adapter_hyper_net_with_bias
        self.one_layer_adapter_hyper_net_with_linear = config.one_layer_adapter_hyper_net_with_linear
        if self.one_layer_adapter_hyper_net:
            self.weight_generator = nn.Parameter(torch.Tensor(self.task_embedding_dim,
                                                              self.input_dim, self.output_dim))
            if self.adapter_hyper_net_with_bias:
                self.bias_generator = nn.Parameter(torch.Tensor(self.task_embedding_dim, self.input_dim))
            else:
                self.register_parameter('bias_generator', None)
            nn.init.normal_(self.weight_generator, std=1e-2)
            if self.bias_generator is not None:
                nn.init.zeros_(self.bias_generator)
        elif self.one_layer_adapter_hyper_net_with_linear:
            self.weight_generator = nn.Sequential(
                linear_layer(self.task_embedding_dim, self.input_dim * self.output_dim))
            self.bias_generator = nn.Sequential(
                linear_layer(self.task_embedding_dim, self.input_dim))
        else:
            self.weight_generator = nn.Sequential(
                linear_layer(self.task_embedding_dim, self.hidden_dim),
                linear_layer(self.hidden_dim, self.input_dim * self.output_dim))
            self.bias_generator = nn.Sequential(
                linear_layer(self.task_embedding_dim, self.hidden_dim),
                linear_layer(self.hidden_dim, self.input_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        if self.one_layer_adapter_hyper_net:
            bias = None
            weight = torch.matmul(task_embedding, self.weight_generator.view(self.task_embedding_dim, -1)
                                  ).view(self.input_dim, self.output_dim)
            if self.adapter_hyper_net_with_bias:
                bias = torch.matmul(task_embedding, self.bias_generator)
        else:
            weight = self.weight_generator(task_embedding).view(self.input_dim, self.output_dim)
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
            nn.ReLU(),
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
        self.set_task_embeddings(config.tasks, config.parametric_task_embedding)
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

    def forward(self, task):
        task_embedding = self.task_to_embeddings[task]
        if self.train_task_embeddings:
            return self.task_hyper_net(task_embedding)
        return task_embedding
