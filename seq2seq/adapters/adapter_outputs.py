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
"""Defines the output class for the adapter layers' parameters."""
from dataclasses import dataclass
import torch


@dataclass
class SamplerOutput:
    """Base class for the base and weights of each adapter."""
    weight: torch.FloatTensor = None
    bias: torch.FloatTensor = None


@dataclass
class LayerNormOutput:
    """Base class for the base and weights of the conditional
    layer norms."""
    weight: torch.FloatTensor = None
    bias: torch.FloatTensor = None


@dataclass
class AdapterOutput:
    """Base class for each adapter weights"""
    up: SamplerOutput = None
    down: SamplerOutput = None
    pre_norm: LayerNormOutput = None
    post_norm: LayerNormOutput = None


@dataclass
class AdapterT5BlockOutput:
    """
    Base class for adapter layer's outputs.
    """
    feed_forward: AdapterOutput = None
    self_attention: AdapterOutput = None
