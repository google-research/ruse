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

"""Implements an Adapter Layer."""
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation

from .adapter_utils import Activations


class Adapter(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.input_dim = config.input_dim
    self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
    self.weight_init_range = config.weight_init_range
    # If reduction factor is not passed we consider default value of 2.
    reduction_factor = config.reduction_factor if config.reduction_factor is not None else 2
    self.down_sample_size = self.input_dim // reduction_factor

    # Construct adapter down sampler module.
    down_sampler_modules = []
    if config.add_layer_norm_before_adapter:
      down_sampler_modules.append(nn.LayerNorm(self.input_dim))
    down_linear = nn.Linear(self.input_dim, self.down_sample_size)
    self.init_linear_layer(down_linear, std=self.weight_init_range)
    down_sampler_modules.append(down_linear)
    down_sampler_modules.append(Activations(config.non_linearity.lower()))
    self.down_sampler = nn.Sequential(*down_sampler_modules)

    # Construct adapter up sampler module.
    self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)
    self.init_linear_layer(self.up_sampler, std=self.weight_init_range)
    if self.add_layer_norm_after_adapter:
      self.post_layer_norm = nn.LayerNorm(self.input_dim)

  def init_linear_layer(self, linear_layer, std):
    """Initializes the linear modules as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)

  def forward(self, x):
    z = self.down_sampler(x)
    output = self.up_sampler(z)
    if self.add_layer_norm_after_adapter:
      output = self.post_layer_norm(output)
    output = output + x
    return output



class MetaAdapter(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.input_dim = config.input_dim
    self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
    self.weight_init_range = config.weight_init_range
    # If reduction factor is not passed we consider default value of 2.
    reduction_factor = config.reduction_factor if config.reduction_factor is not None else 2
    self.down_sample_size = self.input_dim // reduction_factor
    # Construct adapter down sampler module.
    down_sampler_modules = []
    if config.add_layer_norm_before_adapter:
      down_sampler_modules.append(nn.LayerNorm(self.input_dim))
    self.activation_type = config.non_linearity.lower()
    self.down_sampler = nn.Sequential(*down_sampler_modules)
    if self.add_layer_norm_after_adapter:
      self.post_layer_norm = nn.LayerNorm(self.input_dim)

  def init_linear_layer(self, linear_layer, std):
    """Initializes the linear modules as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)

  def forward(self, x, weight_down, bias_down, weight_up, bias_up):
    z = self.down_sampler(x)
    down = F.linear(z, weight=weight_down, bias=bias_down)
    middle = get_activation(self.activation_type)(down)
    output = F.linear(middle, weight=weight_up, bias=bias_up)
    if self.add_layer_norm_after_adapter:
      output = self.post_layer_norm(output)
    output = output + x
    return output
