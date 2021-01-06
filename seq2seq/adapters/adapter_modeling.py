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

"""Implements an Adapter and Hyper-adapter Layers."""
import torch.nn as nn
import torch
from .adapter_utils import Activations, linear_layer, LayerNormHyperNet, TaskHyperNet
from .adapter_outputs import (SamplerOutput, LayerNormOutput,
                              AdapterT5BlockOutput, AdapterOutput)


class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.weight_init_range = config.weight_init_range
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = linear_layer(self.input_dim, self.down_sample_size, std=self.weight_init_range)
        self.up_sampler = linear_layer(self.down_sample_size, self.input_dim, std=self.weight_init_range)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        return self.up_sampler(z)


class AdapterHyperNet(nn.Module):
    """This module generates the weights for the meta adapter layers."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterHyperNet, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_task_embeddings = config.train_task_embeddings
        self.task_embedding_dim = config.projected_task_embedding_dim if \
            config.train_task_embeddings else config.task_embedding_dim
        self.one_layer_adapter_hyper_net = config.one_layer_adapter_hyper_net
        self.adapter_hyper_net_with_bias = config.adapter_hyper_net_with_bias
        self.one_layer_adapter_hyper_net_with_linear = config.one_layer_adapter_hyper_net_with_linear
        # Considers weight and bias parameters for generating adapter weights.
        if self.one_layer_adapter_hyper_net:
            self.weight_generator = nn.Parameter(torch.Tensor(self.task_embedding_dim,
                                                              self.input_dim, self.output_dim))
            if self.adapter_hyper_net_with_bias:
                self.bias_generator = nn.Parameter(torch.Tensor(self.task_embedding_dim, self.input_dim))
            else:
                self.register_parameter('bias_generator', None)
            nn.init.normal_(self.weight_generator, std=1e-2)
            if self.bias_generator is not None:
                nn.init.zeros_(self.bias_generator)
        # Generates the adapter's weight with linear layers.
        elif self.one_layer_adapter_hyper_net_with_linear:
            self.weight_generator = nn.Sequential(
                linear_layer(self.task_embedding_dim, self.input_dim * self.output_dim))
            self.bias_generator = nn.Sequential(
                linear_layer(self.task_embedding_dim, self.input_dim))
        else:
            # Generates the adapter's weight with two linear layers.
            self.weight_generator = nn.Sequential(
                linear_layer(self.task_embedding_dim, self.hidden_dim),
                linear_layer(self.hidden_dim, self.input_dim * self.output_dim))
            self.bias_generator = nn.Sequential(
                linear_layer(self.task_embedding_dim, self.hidden_dim),
                linear_layer(self.hidden_dim, self.input_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        if self.one_layer_adapter_hyper_net:
            bias = None
            weight = torch.matmul(task_embedding, self.weight_generator.view(self.task_embedding_dim, -1)
                                  ).view(self.input_dim, self.output_dim)
            if self.adapter_hyper_net_with_bias:
                bias = torch.matmul(task_embedding, self.bias_generator)
        else:
            weight = self.weight_generator(task_embedding).view(self.input_dim, self.output_dim)
            bias = self.bias_generator(task_embedding).view(-1)
        return weight, bias


class AdapterLayersHyperNet(nn.Module):
    """This module generates the weights for all the meta adapter layers
    given the task embeddings and layer id."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterLayersHyperNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_generator = nn.Sequential(
            linear_layer(config.projected_task_embedding_dim, self.input_dim * self.output_dim))
        self.bias_generator = nn.Sequential(
            linear_layer(config.projected_task_embedding_dim, self.input_dim))

    def forward(self, embeddings):
        weight = self.weight_generator(embeddings).view(self.input_dim, self.output_dim)
        bias = self.bias_generator(embeddings).view(-1)
        return SamplerOutput(weight=weight, bias=bias)


class AdapterLayersHyperNetController(nn.Module):
    """This modules contains the hyper-nets for the feed forward
    and self-attention modules and it generates the adapter's weights and
    layer norm's weights for all the layers of transformers."""

    def __init__(self, config, num_layers=6):
        super(AdapterLayersHyperNetController, self).__init__()
        self.num_layers = num_layers
        self.layer_norm_epsilon = 1e-6
        self.max_position_embeddings = 2
        self.device = config.device
        self.task_embedding_dim = config.task_embedding_dim
        self.layer_id_embeddings = nn.Embedding(self.num_layers,
                                                self.task_embedding_dim).to(self.device)
        self.token_type_embeddings = nn.Embedding(self.max_position_embeddings,
                                                  self.task_embedding_dim).to(self.device)
        config.task_embedding_dim = self.task_embedding_dim * 2
        self.task_hypernet = TaskHyperNet(config)
        config.task_embedding_dim = self.task_embedding_dim
        self.unique_hyper_net_layer_norm = config.unique_hyper_net_layer_norm
        if self.unique_hyper_net_layer_norm:
            self.LayerNorm = nn.LayerNorm(config.projected_task_embedding_dim, eps=self.layer_norm_epsilon)
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        # Defines the adapters hyper-nets.
        self.feed_forward_up_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                       self.input_dim, self.down_sample_size)
        self.feed_forward_down_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                         self.down_sample_size, self.input_dim)
        self.self_attention_up_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                         self.input_dim, self.down_sample_size)
        self.self_attention_down_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                           self.down_sample_size, self.input_dim)
        # Defines the layer norms' hyper net.
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.train_task_embeddings = config.train_task_embeddings
        config.train_task_embeddings = True
        if self.add_layer_norm_before_adapter:
            self.feed_forward_pre_layernorm_hypernet = LayerNormHyperNet(config)
            self.self_attention_pre_layernorm_hypernet = LayerNormHyperNet(config)
        if self.add_layer_norm_after_adapter:
            self.feed_forward_post_layernorm_hypernet = LayerNormHyperNet(config)
            self.self_attention_post_layernorm_hypernet = LayerNormHyperNet(config)
        config.train_task_embeddings = self.train_task_embeddings

    def get_embedding(self, task_embedding, layer_id):
        """Concatenates the task embedding with the embedding for the layer id and
        returns the final joint embedding."""
        layer_id_tensor = torch.tensor([layer_id], dtype=torch.long, device=self.device)
        layer_embedding = self.layer_id_embeddings(layer_id_tensor)
        layer_embedding = layer_embedding.view(-1)
        segment_id = torch.tensor([0, 1], dtype=torch.long, device=self.device)
        embeddings = torch.cat([task_embedding.view(1, -1), layer_embedding.view(1, -1)], axis=0)
        embeddings += self.token_type_embeddings(segment_id)
        embeddings = self.task_hypernet(embeddings.view(-1))
        if self.unique_hyper_net_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        return embeddings

    def forward(self, task_embedding, layer_id):
        embeddings = self.get_embedding(task_embedding, layer_id)
        # Generates the adapters weights in feed-forward and self-attention modules.
        feed_forward_down = self.feed_forward_down_sampler_hyper_net(embeddings)
        feed_forward_up = self.feed_forward_up_sampler_hyper_net(embeddings)
        self_attention_down = self.self_attention_down_sampler_hyper_net(embeddings)
        self_attention_up = self.self_attention_up_sampler_hyper_net(embeddings)
        feed_forward_output = AdapterOutput(up=feed_forward_up, down=feed_forward_down)
        self_attention_output = AdapterOutput(up=self_attention_up, down=self_attention_down)
        # Generates the weights and baises for pre and post layer norms.
        if self.add_layer_norm_before_adapter:
            weight, bias = self.feed_forward_pre_layernorm_hypernet(embeddings)
            feed_forward_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.self_attention_pre_layernorm_hypernet(embeddings)
            self_attention_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
        if self.add_layer_norm_after_adapter:
            weight, bias = self.feed_forward_post_layernorm_hypernet(embeddings)
            feed_forward_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.self_attention_post_layernorm_hypernet(embeddings)
            self_attention_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
        return AdapterT5BlockOutput(feed_forward=feed_forward_output,
                                    self_attention=self_attention_output)
