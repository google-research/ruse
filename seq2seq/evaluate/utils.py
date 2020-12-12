import torch
from typing import Dict
from transformers import is_torch_tpu_available
from transformers.trainer_pt_utils import get_tpu_sampler, SequentialDistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm


def is_world_process_zero():
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be :obj:`True` for one process).
        """
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return True

def _setup_devices():
    if is_torch_tpu_available():
        device = xm.xla_device()
        n_gpu = 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    return device, n_gpu


def get_train_sampler(train_dataset):
        if is_torch_tpu_available():
            return get_tpu_sampler(train_dataset)
        else:
            return RandomSampler(train_dataset)


def get_eval_sampler(eval_dataset):
        if is_torch_tpu_available():
            return SequentialDistributedSampler(eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        else:
            return SequentialSampler(eval_dataset)

class ClassificationCollator:
    def __init__(self, tokenizer, data_args, label_list, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.label_to_id = {v: i for i, v in enumerate(label_list)}
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        # because of padding="longest" this does not work to be done in dataset part.
        batch = self._encode(batch)
        targets = batch["targets"]
        input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
        )
        decoder_input_ids = self._shift_right_t5(labels)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
        return batch, targets

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids

    def _encode(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding="max_length" if self.tpu_num_cores is not None else "longest",  # TPU hack
            return_tensors="pt"
        )
        batch_encoding["targets"] = torch.tensor([self.label_to_id[x["tgt_texts"]] for x in batch])
        return batch_encoding.data
