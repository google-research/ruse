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
"""Defines projection layers for projecting fixed length sentence
representations back to variable length sentence embeddings."""

import abc
from collections import OrderedDict

import torch.nn as nn


class Projection(nn.Module, metaclass=abc.ABCMeta):
    """Projection Layer Abstract Base class."""

    @abc.abstractmethod
    def forward(self, hidden_states):
        """This method implements a projection of the hidden_states.

        Args:
            hidden_states: a Tensor of shape (batch_size x hidden_size x 1)

        Returns:
            A projection of hidden_states to a variable length sentence
            representation of shape (batch_size x seq_length x hidden_size)
        """
        raise NotImplementedError()


class MLP(Projection):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.projection_length = config.projection_length
        self.mlp = nn.Sequential(
            nn.Linear(1, self.projection_length),
            nn.ReLU(),
            nn.Linear(self.projection_length, self.projection_length),
            nn.ReLU(),
            nn.Linear(self.projection_length, self.projection_length)
        )

    def forward(self, hidden_states):
        return self.mlp(hidden_states).transpose(1, 2)


PROJECTION_MAPPING = OrderedDict(
    [
        ("mlp", MLP)
    ]
)


class AutoProjection(nn.Module):
    """A generic projection class to instantiate projection classes."""

    @classmethod
    def get(cls, projection_type, config, *args, **kwargs):
        if projection_type in PROJECTION_MAPPING:
            projection_class = PROJECTION_MAPPING[projection_type]
            return projection_class(config, *args, **kwargs)
        raise ValueError(
            "Unrecognized projection type identifier: {}. Should contain one of {}"
                .format(projection_type, ", ".join(PROJECTION_MAPPING.keys())))
