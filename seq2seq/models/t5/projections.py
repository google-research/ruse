"""Defines projection layers for projecting fixed length sentence
representations back to variable length sentence embeddings."""

import abc
import torch.nn as nn
from collections import OrderedDict

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
