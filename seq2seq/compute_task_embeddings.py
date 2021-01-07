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
"""Computes the task embeddings from pretrained T5 encoder,
by computing the average of encoder's representation over the
whole dataset."""
# usage: python compute_task_embeddings.py configs/task-embedding.json

import sys

import logging
import numpy as np
import os
import torch
from dataclasses import dataclass, field
from third_party.models import T5ForConditionalGeneration, T5Config
from third_party.utils import TaskCollator
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers import TrainingArguments
from typing import Optional, List, Any, Dict, Union

from seq2seq.data import AutoTask

logger = logging.getLogger(__name__)


def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


@dataclass
class ModelArguments:
    """
    Arguments containing pretrained model, config, and tokenizer paths."""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    tasks: Optional[List[str]] = field(
        default="MRPC",
        metadata={"help": "Task name from the list of registered tasks."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )


def _prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], args) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(args.device)
    return inputs


def get_dataloader(dataset: Dataset, args, data_collator) -> DataLoader:
    sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=args.eval_batch_size,
        collate_fn=data_collator,
        drop_last=args.dataloader_drop_last,
        num_workers=args.dataloader_num_workers,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, data_args = parser.parse_args_into_dataclasses()
    create_output_dir(training_args.output_dir)
    set_seed(training_args.seed)
    config = T5Config.from_pretrained(
        model_args.config_name if model_args.config_name else \
            model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else \
            model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir
    )
    model = model.to(training_args.device)
    model.eval()
    dataset_class = AutoTask
    train_datasets = {task: dataset_class.get(task).get_dataset(
        split="train", n_obs=data_args.n_train, add_prefix=False)
        for task in data_args.tasks}

    for train_task, train_dataset in train_datasets.items():
        print("Processing task ", train_task)
        total_avg_hidden_states = 0
        num_examples = 0
        data_collator = TaskCollator(tokenizer, data_args, tpu_num_cores=0)
        dataloader = get_dataloader(train_dataset, training_args, data_collator)
        for step, inputs in enumerate(dataloader):
            inputs = _prepare_inputs(inputs, training_args)
            outputs = model(**inputs, return_dict=True)
            input_mask = inputs['attention_mask']
            hidden_states = outputs.encoder_last_hidden_state
            active_hidden_states = torch.einsum("ijk,ij->ijk", [hidden_states, input_mask])
            avg_hidden_states = active_hidden_states.sum(1) / input_mask.sum(dim=1).view(input_mask.size(0), 1)
            total_avg_hidden_states += avg_hidden_states.sum(dim=0).detach().cpu().numpy()
            num_examples += input_mask.size(0)
        total_avg_hidden_states = total_avg_hidden_states / num_examples
        np.save(os.path.join(training_args.output_dir, '{}.npy'.format(train_task)), total_avg_hidden_states)
