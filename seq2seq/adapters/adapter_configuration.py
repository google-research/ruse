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
  original_ln_before: bool = False
  original_ln_after: bool = True
  residual_before_layer_norm: bool = True
  # adapter_residual_before_ln: bool = False
  add_layer_norm_before: bool = False
  add_layer_norm_after: bool = True  # changed for now was False
  output_adapter: bool = True
  non_linearity: str = "swish"
  reduction_factor: int = 16
  adapter_initializer_range = 1e-2
