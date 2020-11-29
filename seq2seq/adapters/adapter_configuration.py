"""Implements the adapters configuration."""

from dataclasses import dataclass


@dataclass
class AdapterConfig:
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    proposed in https://arxiv.org/abs/1902.00751.
    """
    # TODO: ask original_ln_before, original_ln_after
    original_ln_before: bool = False
    original_ln_after: bool = True
    residual_before_layer_norm: bool = True
    #adapter_residual_before_ln: bool = False
    add_layer_norm_before: bool = False
    add_layer_norm_after: bool = True # changed for now was False
    output_adapter: bool = True
    non_linearity: str = "swish" #"gelu" #"relu" #"gelu" #"swish"
    reduction_factor: int = 16
    init_bert_weights: bool = False #True
    adapter_initializer_range = 1e-2


