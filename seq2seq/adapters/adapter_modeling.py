"""Implements an Adapter block.

Code is adapted from: https://github.com/Adapter-Hub/adapter-transformers/blob/master/\
src/transformers/adapter_modeling.py
Some clean implementation here: https://github.com/hosein-m/TF-Adapter-BERT/blob/master/run_tf_glue_adapter_bert.py
"""
import torch.nn as nn
from .adapter_utils import Activations

class Adapter(nn.Module):
    """Implements a single Adapter block."""

    def __init__(self, model_config, adapter_config):
        super().__init__()
        self.input_size = model_config.d_model
        self.add_layer_norm_before = adapter_config.add_layer_norm_before
        self.add_layer_norm_after = adapter_config.add_layer_norm_after
        self.residual_before_layer_norm = adapter_config.residual_before_layer_norm
        self.eps = model_config.layer_norm_epsilon

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []
        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            seq_list.append(nn.LayerNorm(self.input_size)) # T5LayerNorm(self.input_size, eps=eps))

        # if a downsample size is not passed, we just half the size of the original input
        reduction_factor = adapter_config.reduction_factor if adapter_config.reduction_factor is not None else 2
        self.down_sample_size = self.input_size//reduction_factor
        down_linear=nn.Linear(self.input_size, self.down_sample_size)

        adapter_initializer_range=adapter_config.adapter_initializer_range
        self.init_linear_layer(down_linear, std=adapter_initializer_range)
        seq_list.append(down_linear)
        self.non_linearity = Activations(adapter_config.non_linearity.lower())
        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample.
        # In the forward pass we include the residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        self.adapter_up = nn.Linear(self.down_sample_size, self.input_size)
        self.init_linear_layer(self.adapter_up, std=adapter_initializer_range)

        # If we want to have a layer norm on output, we apply it later after a
        # separate residual connection. This means that we learn a new output layer norm,
        # which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size) # T5LayerNorm(self.input_size, eps=eps)

    def init_linear_layer(self, linear_layer, std):
        """Initializes the linear modules as explained in adapter paper."""
        nn.init.normal_(linear_layer.weight, std=std)
        nn.init.zeros_(linear_layer.bias)

    def forward(self, x): #, residual_input):
        down = self.adapter_down(x)
        up = self.adapter_up(down)
        output = up
        #if self.residual_before_layer_norm:
        #    output = output + residual_input
        #output = output + x
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)
        #if not self.residual_before_layer_norm:
        #    output = output + residual_input
        output = output + x
        #output = self.adapter_norm_after(output)
        return output #, self.adapter_norm_after #, down, up


