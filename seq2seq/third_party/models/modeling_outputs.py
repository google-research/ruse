# Copyright 2020 The HuggingFace Team. All rights reserved.
# Copyright 2020 Google LLC
# Modified from the original HuggingFace version.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput
from typing import List, Optional, Tuple


@dataclass
class RuseBaseModelOutputWithPastAndCrossAttentions(BaseModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to
    speed up sequential decoding).
    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,
         sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If :obj:`past_key_values` is used only the last hidden-state of the
            sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when
        ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with
            each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length,
            embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention
            blocks) that can be used (see:obj:`past_key_values` input) to speed up
            sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when
        ``output_hidden_states=True`` is passed or when
        ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the
            embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial
            embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when
        ``output_attentions=True`` is passed or when
        ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned
        when ``output_attentions=True`` and ``config.add_cross_attention=True``
         is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads,sequence_length, sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the
            attention softmax, used to compute the weighted average in the
            cross-attention heads.
        pooled_enc_hidden_state (:obj:`torch.FloatTensor`, `optional`, returned when ``config.fixed_length_emb=True``
            and shows the encoder pooled hidden state.
    """

    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    pooled_enc_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class RuseSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        pooled_enc_hidden_state (:obj:`torch.FloatTensor`, `optional`, returned when ``config.fixed_length_emb=True``
        and shows the encoder pooled hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    pooled_enc_hidden_state: Optional[torch.FloatTensor] = None
