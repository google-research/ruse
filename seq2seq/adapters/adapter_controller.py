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
"""Implements Adapter Controller, a module that keeps multiple 
layers of Adapters, and controls which adapter layer to use."""
import os

import numpy as np
import torch
import torch.nn as nn

from .adapter_configuration import AdapterConfig, MetaAdapterConfig, ParametricMetaAdapterConfig
from .adapter_modeling import MetaAdapter, Adapter
from .adapter_utils import HyperNetUpSampler, HyperNetDownSampler


class AdapterController(nn.Module):
  """Implements Adapter controller module which controls the logics of
  putting adapter layers within transformer's layers."""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.adapters = nn.ModuleDict(dict())
    self.tasks = config.tasks
    self.adapters = self.construct_adapters(self.tasks)
    self.task_to_adapter = {task: task for task in self.tasks}
    self.add_layer_norm_before_adapter_inside_controller = config.add_layer_norm_before_adapter_inside_controller
    self.add_layer_norm_after_adapter_inside_controller = config.add_layer_norm_after_adapter_inside_controller
    if self.add_layer_norm_before_adapter_inside_controller:
      self.pre_layer_norm = nn.LayerNorm(config.input_dim)
    if self.add_layer_norm_after_adapter_inside_controller:
      self.post_layer_norm = nn.LayerNorm(config.input_dim)

  def set_task_to_adapter_map(self, mapping):
    self.task_to_adapter = mapping

  def get_task(self, task):
    return self.task_to_adapter[task]

  def construct_adapters(self, tasks):
    """
    Constructs adapter layers and adds them to a dictionary for the given
    tasks.
    :param tasks: A list of string containing the task names.
    """
    for task in tasks:
      self.adapters[task] = Adapter(self.config)
    return self.adapters

  def disable_adapters(self, tasks):
    """
    Given a list of tasks, it freezes their corresponding adapter layers'
    parameters.
    :param tasks: Given list of tasks.
    """
    tasks = self.convert_to_list(tasks)
    for task in tasks:
      adapter = self.get_adapter(task)
      for param in adapter.parameters():
        param.requires_grad = False

  def convert_to_list(self, tasks):
    if isinstance(tasks, list):
      return tasks
    else:
      return [tasks]

  def enable_adapters(self, tasks):
    """
    Given a list of tasks, it unfreezes their corresponding adapter layers.
    :param tasks: Given list of tasks.
    """
    tasks = self.convert_to_list(tasks)
    for task in tasks:
      adapter = self.get_adapter(task)
      for param in adapter.parameters():
        param.requires_grad = True

  def get_adapter(self, task):
    """Given a task returns its corresponding adapter layer.
    :param task: Input task name.
    :return: Adapter layer corresponding to the given task.
    """
    return self.adapters[task]

  def call_adapter(self, adapter, inputs, task):
    return adapter(inputs)

  def forward(self, task, inputs):
    """Retrieves the adapter layer corresponding to the given
    task. It freezes the adapter layers for all the other tasks
    and call the selected adapter layer.
    :param task: the name of the current task.
    :param inputs: the inputs to feed in in the adapter layer.
    :return: outputs of the adapter layer."""
    task = self.get_task(task)
    # Enables the adapter layer for the given task.
    self.enable_adapters(task)
    # Disable other adapters.
    other_tasks = [x for x in self.tasks if x != task]
    self.disable_adapters(other_tasks)
    adapter = self.get_adapter(task)
    if self.add_layer_norm_before_adapter_inside_controller:
      inputs = self.pre_layer_norm(inputs)
    outputs = self.call_adapter(adapter, inputs, task)
    if self.add_layer_norm_after_adapter_inside_controller:
      outputs = self.post_layer_norm(outputs)
    return outputs


class MetaAdapterController(AdapterController):
  """Implements Meta Adapter controller module, in which
  the adapter layers' weights are generated from a hyper-network.
  In this case, task-embeddings are fixed, they can be initialized
  from a directory (task_embedding_dir) or if not given, the task
  embeddings will be initialized to random."""

  def __init__(self, config):
    super().__init__(config)
    self.adapters = nn.ModuleDict(dict())
    self.config = config
    self.tasks = config.tasks
    self.adapters = self.construct_adapters(self.tasks)
    self.task_embedding_dir = config.task_embedding_dir
    self.input_dim = config.input_dim
    self.task_to_embeddings = {}
    self.set_task_embeddings(self.tasks, config.task_embedding_dim)
    self.meta_up_sampler = HyperNetUpSampler(config)
    self.meta_down_sampler = HyperNetDownSampler(config)
    self.task_to_adapter = {task: task for task in self.tasks}

  def set_task_embeddings(self, tasks, task_embedding_dim):
    for task in tasks:
      # TODO: fix it.
      if self.task_embedding_dir is not None:
        task_embedding_path = os.path.join(self.task_embedding_dir, task + ".npy")
        self.task_to_embeddings[task] = torch.Tensor(np.load(task_embedding_path)).cuda()
      else:
        self.task_to_embeddings[task] = torch.Tensor(torch.randn(task_embedding_dim)).cuda()

  def construct_adapters(self, tasks):
    """
    Constructs adapter layers and adds them to a dictionary for the given
    tasks.
    :param tasks: A list of string containing task names.
    """
    for task in tasks:
      self.adapters[task] = MetaAdapter(self.config)
    return self.adapters

  def call_adapter(self, adapter, inputs, task):
    task_emb = self.task_to_embeddings[task].cuda()  
    weight_up, bias_up = self.meta_up_sampler(task_emb)
    weight_down, bias_down = self.meta_down_sampler(task_emb)
    return adapter(inputs, weight_down, bias_down, weight_up, bias_up)

  # TODO: this needs to be checked.
  def forward(self, eval_task, inputs):
    """Retrieves the adapter layer corresponding to the given
    task. It freezes the adapter layers for all the other tasks
    and call the selected adapter layer.
    :param task: the name of the current task.
    :param inputs: the inputs to feed in in the adapter layer.
    :return: outputs of the adapter layer."""
    task = self.get_task(eval_task)
    # Enables the adapter layer for the given task.
    self.enable_adapters(task)
    # Disable other adapters.
    other_tasks = [x for x in self.tasks if x != task]
    self.disable_adapters(other_tasks)
    adapter = self.get_adapter(task)
    if self.add_layer_norm_before_adapter_inside_controller:
      inputs = self.pre_layer_norm(inputs)
    outputs = self.call_adapter(adapter, inputs, eval_task)
    if self.add_layer_norm_after_adapter_inside_controller:
      outputs = self.post_layer_norm(outputs)
    return outputs

class MetaParamterizedAdapterController(MetaAdapterController):
  """Implements Meta parameterized Adapter controller module, in which
  the adapter layers' weights are generated from a hyper-network.
  In this case, task-embeddings are parametric, they can be initialized
  from a directory (task_embedding_dir) or if not given, the task
  embeddings will be initialized to random."""

  def __init__(self, config):
    super().__init__(config)
    self.adapters = nn.ModuleDict(dict())
    self.config = config
    self.tasks = config.tasks
    self.adapters = self.construct_adapters(self.tasks)
    self.input_dim = config.input_dim
    self.task_embedding_dir = config.task_embedding_dir
    self.task_to_embeddings = nn.ParameterDict(dict())
    for task in self.tasks:
      if self.task_embedding_dir is not None:
        task_embedding_path = os.path.join(self.task_embedding_dir, task + ".npy")
        task_seed = torch.Tensor(np.load(task_embedding_path)).cuda()
      else:
        task_seed = torch.randn(config.task_embedding_dim).cuda()
      # TODO: fix it.
      self.task_to_embeddings[task] = nn.Parameter(task_seed)
    self.meta_up_sampler = HyperNetUpSampler(config)
    self.meta_down_sampler = HyperNetDownSampler(config)
    self.task_to_adapter = {task: task for task in self.tasks}


class AutoAdapterController(nn.Module):
  """Generic adapter controller class to instantiate different adapter
  controller classes."""

  @classmethod
  def get(cls, config):
    if isinstance(config, ParametricMetaAdapterConfig):
      return MetaParamterizedAdapterController(config)
    elif isinstance(config, MetaAdapterConfig):
      return MetaAdapterController(config)
    elif isinstance(config, AdapterConfig):
      return AdapterController(config)
    raise ValueError("Unrecognized adapter config", config)
