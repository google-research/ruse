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

"""Implements the adapters configuration."""

from dataclasses import dataclass


@dataclass
class AdapterConfig:
  """Implements the adapter configuration proposed by Houlsby et. al, 2019
  proposed in https://arxiv.org/abs/1902.00751."""
  add_layer_norm_before_adapter: bool = False
  add_layer_norm_after_adapter: bool = True
  non_linearity: str = "swish"
  reduction_factor: int = 16
  weight_init_range = 1e-2



class MetaAdapterConfig:
  """Implements the adapter configuration proposed by Houlsby et. al, 2019
  proposed in https://arxiv.org/abs/1902.00751."""
  add_layer_norm_before_adapter: bool = False
  add_layer_norm_after_adapter: bool = True
  non_linearity: str = "swish"
  reduction_factor: int = 16
  weight_init_range = 1e-2
  task_embedding_dim = 64 #768
  hidden_dim = 128
  x_dim = 8  #32
  y_dim = 8 #24
