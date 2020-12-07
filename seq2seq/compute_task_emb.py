"""We compute the task embeddings from pretrained T5 encoder,
by computing the average of encoder's representation over the
whole dataset."""
import os
import sys
import logging
import torch
import numpy as np

from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataset import Dataset
from typing import Optional, List, Any, Dict, Union
from dataclasses import dataclass, field
from transformers import TrainingArguments
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from seq2seq.models import T5ForConditionalGeneration, T5Config
from seq2seq.utils import check_output_dir
from seq2seq.tasks import AutoTask, TaskCollator
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
  """
  Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
  """
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
  check_output_dir(training_args)

  # Setup logging
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
  )
  logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    training_args.local_rank,
    training_args.device,
    training_args.n_gpu,
    bool(training_args.local_rank != -1),
    training_args.fp16,
  )
  logger.info("Training/evaluation parameters %s", training_args)
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
  # TODO: I set add_prefix to False, change if needed.
  dataset_class = AutoTask
  train_datasets = {task: dataset_class.get(task).get_dataset(
    split="train", n_obs=data_args.n_train, add_prefix=False)
                     for task in data_args.tasks}

  for train_task, train_dataset in train_datasets.items():
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
    total_avg_hidden_states = total_avg_hidden_states/num_examples
    print("final shape ", total_avg_hidden_states.shape)
    np.save(os.path.join(training_args.output_dir, '{}.npy'.format(train_task)), total_avg_hidden_states)
