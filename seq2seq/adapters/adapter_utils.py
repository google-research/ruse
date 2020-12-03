import torch.nn as nn
from transformers.activations import get_activation

class Activations(nn.Module):
  """Implementation of various activation function."""

  def __init__(self, activation_type):
    super().__init__()
    self.f = get_activation(activation_type)

  def forward(self, x):
    return self.f(x)
