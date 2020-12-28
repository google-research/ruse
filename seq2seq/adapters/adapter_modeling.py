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

"""Implements an Adapter Layer and Meta Adapter layer."""
import torch.nn as nn

from .adapter_utils import Activations, linear_layer


class Adapter(nn.Module):
  """Conventional Adapter layer, in which the weights of up and down sampler modules
  are parameters and are optimized."""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.input_dim = config.input_dim
    self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
    self.weight_init_range = config.weight_init_range
    self.down_sample_size = self.input_dim // config.reduction_factor
    down_sampler_modules = []
    if config.add_layer_norm_before_adapter:
      down_sampler_modules.append(nn.LayerNorm(self.input_dim))
    down_linear = linear_layer(self.input_dim, self.down_sample_size, std=self.weight_init_range)
    down_sampler_modules.append(down_linear)
    down_sampler_modules.append(Activations(config.non_linearity.lower()))
    self.down_sampler = nn.Sequential(*down_sampler_modules)
    self.up_sampler = linear_layer(self.down_sample_size, self.input_dim, std=self.weight_init_range)
    if self.add_layer_norm_after_adapter:
      self.post_layer_norm = nn.LayerNorm(self.input_dim)

  def forward(self, x):
    z = self.down_sampler(x)
    output = self.up_sampler(z)
    if self.add_layer_norm_after_adapter:
      output = self.post_layer_norm(output)
    output = output + x
    return output
