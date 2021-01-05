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
"""Defines the trainer module to train a classifier on top of embeddings."""

import math
import torch

import torch.nn as nn
from transformers import AdamW
from third_party.models import T5ForConditionalGeneration

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import is_torch_tpu_available

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl


class ClassificationNet(nn.Module):
    def __init__(self, args, config):
        super(ClassificationNet, self).__init__()
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.num_classes = args.num_classes
        self.encoder = T5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            from_tf=".ckpt" in args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
        )
        self.encoder = self.freeze(self.encoder)
        self.encoder = self.freeze(self.encoder)
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        self.device = args.device

    def freeze(self, model):
        """Freezes the given model's parameters."""
        for p in model.parameters():
            p.requires_grad = False
        return model

    def forward(self, inputs):
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
        embedding = (self.encoder(**inputs)[4]).squeeze()
        output = self.classifier(embedding)
        return output


class Trainer(nn.Module):
    """This class trains a classifier on top of pretrained embeddings."""

    def __init__(self, args, config, train_dataloader, eval_dataloader):
        super(Trainer, self).__init__()
        self.learning_rate = args.learning_rate
        self.num_train_epochs = args.num_train_epochs
        self.model = ClassificationNet(args, config)
        self.device = args.device
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        if args.n_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.n_gpus = args.n_gpus

    def train(self):
        """Trains the classifier on the training embeddings."""
        num_train_epochs = math.ceil(self.num_train_epochs)
        self.model.train()
        self.model.zero_grad()
        for epoch in range(num_train_epochs):
            if isinstance(self.train_dataloader, DataLoader) and \
                    isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)
            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(self.train_dataloader, [self.device]).per_device_loader(
                    self.device
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = self.train_dataloader
            for step, (inputs, targets) in enumerate(epoch_iterator):
                output = self.model(inputs)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_fn(output, targets)
                if self.n_gpus > 1:
                    loss = loss.mean()
                loss.backward()

                if is_torch_tpu_available():
                    xm.optimizer_step(self.optimizer)
                else:
                    self.optimizer.step()

                self.model.zero_grad()

    def evaluate(self):
        """Evaluates the classifier accuracy on the given inputs."""
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(self.eval_dataloader):
                output = self.model(inputs)
                targets = targets.to(self.device)
                predictions = output.data.max(1)[1]
                correct += predictions.long().eq(targets.data.long()).cpu().sum()
        num_samples = len(self.eval_dataloader.dataset)
        accuracy = correct / num_samples
        return accuracy
