"""Implements an Adapter block.

Code is adapted from: https://github.com/Adapter-Hub/adapter-transformers/blob/master/\
src/transformers/adapter_modeling.py
Some clean implementation here: https://github.com/hosein-m/TF-Adapter-BERT/blob/master/run_tf_glue_adapter_bert.py
"""

import torch.nn as nn
from .adapter_utils import Activations
from transformers.modeling_t5 import T5LayerNorm

class Adapter(nn.Module):
    """
    Implementation of a single Adapter block.
    """

    def __init__(self, input_size, config, eps):
        super().__init__()
        self.input_size = input_size
        self.add_layer_norm_before = config.add_layer_norm_before
        self.add_layer_norm_after = config.add_layer_norm_after
        self.residual_before_layer_norm = config.residual_before_layer_norm
        self.eps = eps

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []
        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            seq_list.append(nn.LayerNorm(self.input_size)) # T5LayerNorm(self.input_size, eps=eps))

        # if a downsample size is not passed, we just half the size of the original input
        reduction_factor = config.reduction_factor if config.reduction_factor is not None else 2
        self.down_sample_size = self.input_size//reduction_factor
        down_linear=nn.Linear(self.input_size, self.down_sample_size)

        ### INIT
        adapter_initializer_range=config.adapter_initializer_range #1e-2
        nn.init.normal_(down_linear.weight, std=adapter_initializer_range)
        nn.init.zeros_(down_linear.bias)


        seq_list.append(down_linear)
        self.non_linearity = Activations(config.non_linearity.lower())
        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample.
        # In the forward pass we include the residual connection
        self.adapter_down = nn.Sequential(*seq_list)



        # Up projection to input size
        self.adapter_up = nn.Linear(self.down_sample_size, self.input_size)


        ## initialization.
        nn.init.normal_(self.adapter_up.weight, std=adapter_initializer_range)
        nn.init.zeros_(self.adapter_up.bias)



        # If we want to have a layer norm on output, we apply it later after a
        # separate residual connection. This means that we learn a new output layer norm,
        # which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size) # T5LayerNorm(self.input_size, eps=eps)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if config.init_bert_weights:
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)

    def forward(self, x): #, residual_input):
        down = self.adapter_down(x)
        up = self.adapter_up(down)
        output = up

        # todo add brief documentation what that means
        #if self.residual_before_layer_norm:
        #    output = output + residual_input

        #output = output + x

        # todo add brief documentation what that means
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        # todo add brief documentation what that means
        #if not self.residual_before_layer_norm:
        #    output = output + residual_input
        output = output + x

        #output = self.adapter_norm_after(output)

        return output #, self.adapter_norm_after #, down, up


    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # TODO I set the std to default 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
