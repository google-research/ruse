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

"""Implements the adapters' configurations."""

from collections import OrderedDict
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class AdapterConfig(object):
  """Implements the adapter configuration proposed by Houlsby et. al, 2019
  proposed in https://arxiv.org/abs/1902.00751."""
  add_layer_norm_before_adapter: bool = False
  add_layer_norm_after_adapter: bool = True
  non_linearity: str = "swish"
  reduction_factor: int = 16
  weight_init_range = 1e-2


class MetaAdapterConfig(AdapterConfig):
  """Implements Meta adapter in which a hyper-network generates the parameters of
   adapter layers. Task embeddings are fixed in this case."""
  task_embedding_dim = 512
  task_embedding_dir = None
  hidden_dim = 128
  train_task_embeddings = False
  projected_task_embedding_dim = 64


class ParametricMetaAdapterConfig(AdapterConfig):
  """Implements meta adapter configuration, in which a hyper-network generates the
  parameters of adapter layers. Task embeddings are parameters in this case."""
  hidden_dim = 128
  task_embedding_dir = None
  task_embedding_dim = 64
  train_task_embeddings = False
  projected_task_embedding_dim = 64


ADAPTER_CONFFIG_MAPPING = OrderedDict(
  [("adapter", AdapterConfig),
   ("meta-adapter", MetaAdapterConfig),
   ("parametric-meta-adapter", ParametricMetaAdapterConfig)])


class AutoAdapterConfig(nn.Module):
  """Generic Adapter config class to instantiate different adapter configs."""

  @classmethod
  def get(cls, config_name: str):
    if config_name in ADAPTER_CONFFIG_MAPPING:
      return ADAPTER_CONFFIG_MAPPING[config_name]()
    raise ValueError(
      "Unrecognized adapter config type identifier: {}. Should contain one of {}"
        .format(config_name, ", ".join(ADAPTER_CONFFIG_MAPPING.keys())))
