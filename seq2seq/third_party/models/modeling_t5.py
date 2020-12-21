# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model. """

import copy
import warnings

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_t5 import (T5PreTrainedModel, T5LayerNorm, T5Block,
                                      T5DenseReluDense, T5Attention, T5LayerCrossAttention)
from transformers.utils import logging

from seq2seq.adapters import AutoAdapterController
from .modeling_outputs import RuseBaseModelOutputWithPastAndCrossAttentions, RuseSeq2SeqLMOutput
from seq2seq.models.poolings import AutoPooling
from seq2seq.models.projections import AutoProjection
from seq2seq.adapters import MetaAdapterConfig, ParametricMetaAdapterConfig
from seq2seq.adapters import TaskEmbeddingController

logger = logging.get_logger(__name__)


class T5LayerFF(nn.Module):
  def __init__(self, config, adapter_config=None):
    super().__init__()
    self.DenseReluDense = T5DenseReluDense(config)
    self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
    self.dropout = nn.Dropout(config.dropout_rate)
    self.train_adapters = config.train_adapters
    if self.train_adapters:
      self.adapter_controller = AutoAdapterController.get(adapter_config)
      self.is_meta_adapter = True if isinstance(adapter_config, (MetaAdapterConfig, ParametricMetaAdapterConfig)) else False

  def forward(self, hidden_states, task=None, task_embedding=None):
    norm_x = self.layer_norm(hidden_states)
    y = self.DenseReluDense(norm_x)
    if self.train_adapters:
      if not self.is_meta_adapter:
        y = self.adapter_controller(task, y)
      else:
        y = self.adapter_controller(task_embedding, y)
    layer_output = hidden_states + self.dropout(y)
    return layer_output


class T5LayerSelfAttention(nn.Module):
  def __init__(self, config, has_relative_attention_bias=False, adapter_config=None):
    super().__init__()
    self.SelfAttention = T5Attention(
      config, has_relative_attention_bias=has_relative_attention_bias, is_bidirectional=not config.is_decoder
    )
    self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
    self.dropout = nn.Dropout(config.dropout_rate)
    self.train_adapters = config.train_adapters
    if self.train_adapters:
      self.adapter_controller = AutoAdapterController.get(adapter_config)
      self.is_meta_adapter = True if isinstance(adapter_config,
                                                (MetaAdapterConfig, ParametricMetaAdapterConfig)) else False

  def forward(
      self,
      hidden_states,
      attention_mask=None,
      position_bias=None,
      head_mask=None,
      past_key_value=None,
      use_cache=False,
      output_attentions=False,
      task=None,
      task_embedding=None
  ):
    norm_x = self.layer_norm(hidden_states)
    attention_output = self.SelfAttention(
      norm_x,
      mask=attention_mask,
      position_bias=position_bias,
      head_mask=head_mask,
      past_key_value=past_key_value,
      use_cache=use_cache,
      output_attentions=output_attentions,
    )
    y = attention_output[0]
    if self.train_adapters:
      if not self.is_meta_adapter:
        y = self.adapter_controller(task, y)
      else:
        y = self.adapter_controller(task_embedding, y)
    layer_output = hidden_states + self.dropout(y)
    outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
    return outputs


class T5Block(nn.Module):
  def __init__(self, config, has_relative_attention_bias=False, adapter_config=None):
    super().__init__()
    self.adapter_config = adapter_config
    self.is_decoder = config.is_decoder
    self.layer = nn.ModuleList()
    self.layer.append(T5LayerSelfAttention(config, \
                                           has_relative_attention_bias=has_relative_attention_bias,
                                           adapter_config=self.adapter_config))
    if self.is_decoder:
      self.layer.append(T5LayerCrossAttention(config, \
                                              has_relative_attention_bias=has_relative_attention_bias))
    self.layer.append(T5LayerFF(config, self.adapter_config))

  def forward(
      self,
      hidden_states,
      attention_mask=None,
      position_bias=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None,
      encoder_decoder_position_bias=None,
      head_mask=None,
      past_key_value=None,
      use_cache=False,
      output_attentions=False,
      return_dict=False,
      task=None,
      task_embedding=None
  ):
    if past_key_value is not None:
      assert self.is_decoder, "Only decoder can use `past_key_values`"
      expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

      error_message = "There should be {} past states. 2 (past / key)\
            for self attention.{} Got {} past key / value states".format(
        expected_num_past_key_values, "2 (past / key) for cross \
                attention" if expected_num_past_key_values == 4 else "", \
        len(past_key_value),
      )
      assert len(past_key_value) == expected_num_past_key_values, \
        error_message

      self_attn_past_key_value = past_key_value[:2]
      cross_attn_past_key_value = past_key_value[2:]
    else:
      self_attn_past_key_value, cross_attn_past_key_value = None, None

    self_attention_outputs = self.layer[0](
      hidden_states,
      attention_mask=attention_mask,
      position_bias=position_bias,
      head_mask=head_mask,
      past_key_value=self_attn_past_key_value,
      use_cache=use_cache,
      output_attentions=output_attentions,
      task=task,
      task_embedding=task_embedding
    )
    hidden_states, present_key_value_state = self_attention_outputs[:2]
    # Keep self-attention outputs and relative position weights
    attention_outputs = self_attention_outputs[2:]

    do_cross_attention = self.is_decoder and encoder_hidden_states is not None
    if do_cross_attention:
      # the actual query length is unknown for cross attention
      # if using past key value states. Need to inject it here
      if present_key_value_state is not None:
        query_length = present_key_value_state[0].shape[2]
      else:
        query_length = None

      # TODO(rabeeh): We do not for now add adapters to cross-attention
      #   layers, might to at some point.
      cross_attention_outputs = self.layer[1](
        hidden_states,
        kv=encoder_hidden_states,
        attention_mask=encoder_attention_mask,
        position_bias=encoder_decoder_position_bias,
        head_mask=head_mask,
        past_key_value=cross_attn_past_key_value,
        query_length=query_length,
        use_cache=use_cache,
        output_attentions=output_attentions,
      )
      hidden_states = cross_attention_outputs[0]
      # Combine self attn and cross attn key value states
      if present_key_value_state is not None:
        present_key_value_state = present_key_value_state + \
                                  cross_attention_outputs[1]

      # Keep cross-attention outputs and relative position weights
      attention_outputs = attention_outputs + cross_attention_outputs[2:]

    # Apply Feed Forward layer
    hidden_states = self.layer[-1](hidden_states, task=task, task_embedding=task_embedding)
    outputs = (hidden_states,)

    outputs = outputs + (present_key_value_state,) + attention_outputs
    return outputs  # hidden-states, present_key_value_states,
    # (self-attention weights), (self-attention position bias),
    # (cross-attention weights), (cross-attention position bias)


class T5Stack(T5PreTrainedModel):
  def __init__(self, config, embed_tokens=None, adapter_config=None):
    super().__init__(config)
    self.adapter_config = adapter_config
    self.embed_tokens = embed_tokens
    self.is_decoder = config.is_decoder

    self.block = nn.ModuleList(
      [T5Block(config, has_relative_attention_bias=bool(i == 0), adapter_config=self.adapter_config)
       for i in range(config.num_layers)]
    )
    self.final_layer_norm = T5LayerNorm(config.d_model,
                                        eps=config.layer_norm_epsilon)
    self.dropout = nn.Dropout(config.dropout_rate)

    self.fixed_length_emb = config.fixed_length_emb
    self.concat_projection_token = config.concat_projection_token
    if self.fixed_length_emb and not self.is_decoder:
      self.only_projection_bottleneck = config.only_projection_bottleneck
      self.encoder_pooler = AutoPooling.get(config.encoder_pooling, config)
      if not self.concat_projection_token:
        self.encoder_projection = AutoProjection.get(config.encoder_projection, config)

    self.init_weights()

  def get_input_embeddings(self):
    return self.embed_tokens

  def get_output_embeddings(self):
    return self.embed_tokens

  def set_input_embeddings(self, new_embeddings):
    self.embed_tokens = new_embeddings

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None,
      inputs_embeds=None,
      head_mask=None,
      past_key_values=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
      task=None,
      task_embedding=None
  ):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
      output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
      err_msg_prefix = "decoder_" if self.is_decoder else ""
      raise ValueError(
        f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
      )
    elif input_ids is not None:
      input_shape = input_ids.size()
      input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
      input_shape = inputs_embeds.size()[:-1]
    else:
      err_msg_prefix = "decoder_" if self.is_decoder else ""
      raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

    if inputs_embeds is None:
      assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
      inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = input_shape

    # required mask seq length can be calculated via length of past
    mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

    if use_cache is True:
      assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
        self
      )

    if attention_mask is None:
      attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
    if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
      encoder_seq_length = encoder_hidden_states.shape[1]
      encoder_attention_mask = torch.ones(
        batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
      )

    # initialize past_key_values with `None` if past does not exist
    if past_key_values is None:
      past_key_values = [None] * len(self.block)

    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

    if self.is_decoder and encoder_attention_mask is not None:
      encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
      encoder_extended_attention_mask = None

    # Prepare head mask if needed
    head_mask = self.get_head_mask(head_mask, self.config.num_layers)
    present_key_value_states = () if use_cache else None
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and self.is_decoder) else None
    position_bias = None
    encoder_decoder_position_bias = None
    hidden_states = self.dropout(inputs_embeds)

    for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
      if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

      layer_outputs = layer_module(
        hidden_states,
        attention_mask=extended_attention_mask,
        position_bias=position_bias,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        encoder_decoder_position_bias=encoder_decoder_position_bias,
        head_mask=head_mask[i],
        past_key_value=past_key_value,
        use_cache=use_cache,
        output_attentions=output_attentions,
        task=task,
        task_embedding=task_embedding
      )
      # layer_outputs is a tuple with:
      # hidden-states, key-value-states, (self-attention weights),
      # (self-attention position bias), (cross-attention weights),
      # (cross-attention position bias)
      hidden_states, present_key_value_state = layer_outputs[:2]

      if i == 0:
        # We share the position biases between the layers - the first layer store them
        # layer_outputs = hidden-states, key-value-states (self-attention weights),
        # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
        position_bias = layer_outputs[3 if output_attentions else 2]
        if self.is_decoder and encoder_hidden_states is not None:
          encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 3]
      # append next layer key value states
      if use_cache:
        present_key_value_states = present_key_value_states + (present_key_value_state,)

      if output_attentions:
        all_attentions = all_attentions + (layer_outputs[2],)
        if self.is_decoder:
          all_cross_attentions = all_cross_attentions + (layer_outputs[4 if i == 0 else 3],)

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    # Learns fixed length embeddings and project them back to the
    # variable length embeddings.
    pooled_hidden_state = None
    if self.fixed_length_emb and not self.is_decoder:
      pooled_hidden_state = self.encoder_pooler(hidden_states, attention_mask)
      if self.concat_projection_token:
        projection_hidden_states = pooled_hidden_state
        projection_hidden_states = projection_hidden_states.transpose(1, 2)
      else:
        projection_hidden_states = self.encoder_projection(pooled_hidden_state)
      if self.only_projection_bottleneck:
        hidden_states = projection_hidden_states
      else:
        hidden_states = torch.cat((projection_hidden_states, hidden_states), dim=1)

    # Add last layer
    if output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
      return tuple(
        v
        for v in [
          hidden_states,
          present_key_value_states,
          all_hidden_states,
          all_attentions,
          all_cross_attentions,
          pooled_hidden_state
        ]
        if v is not None
      )
    return RuseBaseModelOutputWithPastAndCrossAttentions(
      last_hidden_state=hidden_states,
      past_key_values=present_key_value_states,
      hidden_states=all_hidden_states,
      attentions=all_attentions,
      cross_attentions=all_cross_attentions,
      pooled_enc_hidden_state=pooled_hidden_state,
    )


class T5ForConditionalGeneration(T5PreTrainedModel):
  authorized_missing_keys = [r"encoder\.embed_tokens\.weight",
                             r"decoder\.embed_tokens\.weight", r"lm_head\.weight"]

  def __init__(self, config, adapter_config=None):
    super().__init__(config)

    # Computes the task-embeddings.
    self.train_adapters = config.train_adapters 
    if config.train_adapters:
      self.task_embedding_controller = TaskEmbeddingController(adapter_config)

    self.adapter_config = adapter_config
    self.model_dim = config.d_model
    self.shared = nn.Embedding(config.vocab_size, config.d_model)
    encoder_config = copy.deepcopy(config)
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False
    if config.train_adapters:
      encoder_config.train_adapters = True
    self.encoder = T5Stack(encoder_config, self.shared, adapter_config=adapter_config)
    decoder_config = copy.deepcopy(config)
    decoder_config.is_decoder = True
    decoder_config.is_encoder_decoder = False
    decoder_config.num_layers = config.num_decoder_layers
    if config.train_adapters:
      decoder_config.train_adapters = adapter_config.add_adapters_in_decoder
    self.decoder = T5Stack(decoder_config, self.shared, adapter_config=adapter_config)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    self.fixed_length_emb = config.fixed_length_emb
    self.only_projection_bottleneck = config.only_projection_bottleneck
    self.concat_projection_token = config.concat_projection_token
    self.init_weights()

  def get_input_embeddings(self):
    return self.shared

  def set_input_embeddings(self, new_embeddings):
    self.shared = new_embeddings
    self.encoder.set_input_embeddings(new_embeddings)
    self.decoder.set_input_embeddings(new_embeddings)

  def get_output_embeddings(self):
    return self.lm_head

  def get_encoder(self):
    return self.encoder

  def get_decoder(self):
    return self.decoder

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      decoder_input_ids=None,
      decoder_attention_mask=None,
      encoder_outputs=None,
      past_key_values=None,
      head_mask=None,
      inputs_embeds=None,
      decoder_inputs_embeds=None,
      labels=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
      task=None,
      task_embedding=None,
      **kwargs,
  ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`,
    `optional`):
        Labels for computing the sequence classification/regression loss.
        Indices should be in :obj:`[-100, 0, ...,
        config.vocab_size - 1]`. All labels set to ``-100`` are ignored
        (masked), the loss is only computed for
        labels in ``[0, ..., config.vocab_size]``
    kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
        Used to hide legacy arguments that have been deprecated.

    Returns:

    Examples::

        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small',
        return_dict=True)

        >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1>
        park', return_tensors='pt').input_ids
        >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the
        <extra_id_2> </s>', return_tensors='pt').input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> input_ids = tokenizer("summarize: studies have shown that owning
        a dog is good for you ", return_tensors="pt").input_ids# Batch size 1
        >>> outputs = model.generate(input_ids)
    """
    if "lm_labels" in kwargs:
      warnings.warn(
        "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
        FutureWarning,
      )
      labels = kwargs.pop("lm_labels")
    if "decoder_past_key_value_states" in kwargs:
      warnings.warn(
        "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
        FutureWarning,
      )
      past_key_values = kwargs.pop("decoder_past_key_value_states")
    if "decoder_past_key_values" in kwargs:
      warnings.warn(
        "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
        FutureWarning,
      )
      past_key_values = kwargs.pop("decoder_past_key_values")
    assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
      # Convert encoder inputs in embeddings if needed
      encoder_outputs = self.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        task=task,
        task_embedding=self.task_embedding_controller(task) if self.train_adapters else None 
      )
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
      # TODO(rabeeh): from what I see it does not go here.
      encoder_outputs = BaseModelOutput(
        last_hidden_state=encoder_outputs[0],
        hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
      )
    hidden_states = encoder_outputs[0]
    if self.fixed_length_emb:
      # Appends the attention mask for the projection of fixed length embeddings
      # to the attention mask of hidden states.
      if self.concat_projection_token:
        projection_length = 1
      else:
        projection_length = self.config.projection_length
      attention_mask_projection = torch.ones(hidden_states.shape[0],
                                             projection_length, device=attention_mask.device, dtype=torch.long)
      if self.only_projection_bottleneck:
        attention_mask = attention_mask_projection
      else:
        attention_mask = torch.cat((attention_mask_projection,
                                    attention_mask), dim=1)

    if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
      # get decoder inputs from shifting lm labels to the right
      decoder_input_ids = self._shift_right(labels)

    # If decoding with past key value states, only the last tokens
    # should be given as an input
    if past_key_values is not None:
      assert labels is None, "Decoder should not use cached key value states when training."
      if decoder_input_ids is not None:
        decoder_input_ids = decoder_input_ids[:, -1:]
      if decoder_inputs_embeds is not None:
        decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

    # Decode
    decoder_outputs = self.decoder(
      input_ids=decoder_input_ids,
      attention_mask=decoder_attention_mask,
      inputs_embeds=decoder_inputs_embeds,
      past_key_values=past_key_values,
      encoder_hidden_states=hidden_states,
      encoder_attention_mask=attention_mask,
      head_mask=head_mask,
      use_cache=use_cache,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
      task=task,
      task_embedding=self.task_embedding_controller(task) if self.train_adapters else None
    )

    sequence_output = decoder_outputs[0]
    # Rescale output before projecting on vocab
    # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
    sequence_output = sequence_output * (self.model_dim ** -0.5)
    lm_logits = self.lm_head(sequence_output)

    loss = None
    if labels is not None:
      loss_fct = CrossEntropyLoss(ignore_index=-100)
      loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
      # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

    if not return_dict:
      output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
      return ((loss,) + output) if loss is not None else output

    return RuseSeq2SeqLMOutput(
      loss=loss,
      logits=lm_logits,
      past_key_values=decoder_outputs.past_key_values,
      decoder_hidden_states=decoder_outputs.hidden_states,
      decoder_attentions=decoder_outputs.attentions,
      cross_attentions=decoder_outputs.cross_attentions,
      encoder_last_hidden_state=encoder_outputs.last_hidden_state,
      encoder_hidden_states=encoder_outputs.hidden_states,
      encoder_attentions=encoder_outputs.attentions,
      pooled_enc_hidden_state=encoder_outputs.pooled_enc_hidden_state,
    )

  def prepare_inputs_for_generation(
      self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
  ):
    # cut decoder_input_ids if past is used
    if past is not None:
      input_ids = input_ids[:, -1:]
    return {
      "decoder_input_ids": input_ids,
      "past_key_values": past,
      "encoder_outputs": encoder_outputs,
      "attention_mask": attention_mask,
      "use_cache": use_cache,
      "task": kwargs["task"],
      "task_embedding": kwargs["task_embedding"] #self.task_embedding_controller(kwargs["task"])
    }

  def _reorder_cache(self, past, beam_idx):
    # if decoder past is not included in output
    # speedy decoding is disabled and no need to reorder
    if past is None:
      logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
      return past

    reordered_decoder_past = ()
    for layer_past_states in past:
      # get the correct batch idx from layer past batch dim
      # batch dim of `past` is at 2nd position
      reordered_layer_past_states = ()
      for layer_past_state in layer_past_states:
        # need to set correct `past` for each of the four key / value states
        reordered_layer_past_states = reordered_layer_past_states + (
          layer_past_state.index_select(0, beam_idx),
        )

      assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
      assert len(reordered_layer_past_states) == len(layer_past_states)

      reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
    return reordered_decoder_past
