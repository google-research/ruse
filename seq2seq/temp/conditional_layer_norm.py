import torch
import torch.nn as nn 
input_dim  = 129
input = torch.rand(129)
a = nn.LayerNorm(input_dim)
print(a(input))
print(a.weight.shape, a.bias.shape)
print(a.normalized_shape)
weight = torch.rand(129)
bias = torch.randn(129)
normalized_shape=(129,)
b = torch.nn.functional.layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=1e-05)
print(b.shape)
