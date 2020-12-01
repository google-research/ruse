from collections import OrderedDict
from typing import Callable, Dict, Mapping, List
import torch
from dataclasses import dataclass
import datasets
import abc
from transformers import T5Tokenizer
from torch.utils.data.dataloader import DataLoader
import numpy as np 

def compute_task_max_decoding_length(word_list):
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    max_len = 0
    for word in word_list:
        ids = tokenizer.encode(word)
        max_len = max(max_len, len(ids))
    return max_len


@dataclass
class Task:
    """
    Defines the task meta data.
    Args:
        name: (:obj:`str`):
            Defines the task name.
        category (:obj:`str`):
            Defines the category the task belongs to like classification, question answering, etc.
    """
    name: str
    category: str


# TODO: max length per task needs to be corrected
class AbstractTaskDataset(abc.ABC):
    task_specific_config: Dict = NotImplemented
    task: Task = NotImplemented
    preprocessor: Callable = NotImplemented
    split_to_data_split: Mapping[str, str] = \
        {"train":"train", "validation":"validation", "test":"test"}

    def get_sampled_split(self, split, n_obs=None):
        split = self.split_to_data_split[split]
        if n_obs is not None:
            split = split + "[:{}]".format(n_obs)
        return split

    def load_dataset(self, split):
        return datasets.load_dataset(self.task.name, split=split)

    def get_dataset(self, split, n_obs=None):
        split = self.get_sampled_split(split, n_obs)
        dataset = self.load_dataset(split=split)
        dataset = dataset.map(self.preprocessor, remove_columns=dataset.column_names)
        return dataset


class SquadTaskDataset(AbstractTaskDataset):
    task = Task(name="squad", category="question_answering")
    split_to_data_split = {"train": "train", "validation": "validation"}

    def preprocessor(self, example):
        return {"src_texts": "question: {0} context: {1} ".format(
            example["question"], example["context"]),
            "tgt_texts": example["answers"]["text"][0]}


class IMDBTaskDataset(AbstractTaskDataset):
    task = Task(name="imdb", category="classification")
    split_to_data_split = {"train": "train", "validation": "test"}
    label_list = ["pos", "neg"]
    task_specific_config = {'max_length': 3}

    def preprocessor(self, example):
        return {"src_texts": "imdb: " + example["text"],
                "tgt_texts": str(example["label"])}


class BoolQTaskDataset(AbstractTaskDataset):
    task_specific_config = {'max_length': 3}
    task = Task(name="boolq", category="classification")
    split_to_data_split = {"train": "train", "validation": "validation", "test":"validation"}
    label_list=["0", "1"]

    def preprocessor(self, example):
        return {"src_texts": "Boolq question: {} passage: {}: ".format(example["question"], example["passage"]),
                "tgt_texts": str(example["answer"])}


class SNLITaskDataset(AbstractTaskDataset):
    task_specific_config = {'max_length': 5, 'num_beams': 4}
    task = Task(name="snli", category="classification")
    task_specific_config = {'max_length': 3}
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}
    label_list = ["0", "1", "2"]

    def preprocessor(self, example):
        return {"src_texts": "SNLI premise {} hypothesis {}: ".format(example["premise"], example["hypothesis"]),
                "tgt_texts": str(example["label"])}


# TODO: the class should get the name of pairs as a argument.
# and register each class with its own arguments.
class IWSLT2017RONL(AbstractTaskDataset):
    task = Task(name="iwslt2017-ro-nl", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-nl"

    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", 'iwslt2017-ro-nl', split=split)

    def preprocessor(self, example):
        return {"src_texts": "Translate Romanian to Dutch:  {}".format(example['translation']["ro"]),
                "tgt_texts": str(example['translation']["nl"])}



class IWSLT2017ENNL(AbstractTaskDataset):
    task = Task(name="iwslt2017-en-nl", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"en-nl"

    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", 'iwslt2017-en-ko', split=split)

    def preprocessor(self, example):
        return {"src_texts": "Translate English to Dutch:  {}".format(example['translation']["en"]),
                "tgt_texts": str(example['translation']["nl"])}



class WMT16ENROTaskDataset(AbstractTaskDataset):
    task = Task(name="wmt16-en-ro", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example):
        return {"src_texts": "Translate English to Romanian:  {}".format(example['translation']["en"]),
                "tgt_texts": str(example['translation']["ro"])}


class WMT16ROENTaskDataset(AbstractTaskDataset):
    task = Task(name="wmt16-ro-en", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example):
        return {"src_texts": "Translate Romanian to English:  {}".format(example['translation']["ro"]),
                "tgt_texts": str(example['translation']["en"])}


class WMT16ENCSTaskDataset(AbstractTaskDataset):
    task = Task(name="wmt16-en-cs", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"cs-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example):
        return {"src_texts": "Translate English to Czech:  {}".format(example['translation']["en"]),
                "tgt_texts": str(example['translation']["cs"])}


class WMT16ENFITaskDataset(AbstractTaskDataset):
    task = Task(name="wmt16-en-fi", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"fi-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example):
        return {"src_texts": "Translate English to Finnish:  {}".format(example['translation']["en"]),
                "tgt_texts": str(example['translation']["fi"])}


class WMT14HIENTaskDataset(AbstractTaskDataset):
    task = Task(name="wmt14-hi-en", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"hi-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt14", self.pair, split=split)

    def preprocessor(self, example):
        return {"src_texts": "Translate English to Romanian:  {}".format(example['translation']["hi"]),
                "tgt_texts": str(example['translation']["en"])}


class TRECTaskDataset(AbstractTaskDataset):
    task = Task(name="trec", category="classification")
    label_list=["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset("trec",  split=split)

    def preprocessor(self, example):
        return {"src_texts": "Trec sentence :  {}".format(example['text']),
                "tgt_texts": str(example['label-coarse'])}


class YelpPolarityTaskDataset(AbstractTaskDataset):
    task = Task(name="yelp_polarity", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train", "validation": "test", "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset("yelp_polarity",  split=split)

    def preprocessor(self, example):
        return {"src_texts": "Yelp Polarity sentence :  {}".format(example['text']),
                "tgt_texts": str(example['label'])}


class ScitailTaskDataset(AbstractTaskDataset):
    task = Task(name="scitail", category="classification")
    label_list = ["entailment", "neutral"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        dataset = datasets.load_dataset("scitail", "snli_format", split=split)
        return dataset

    def preprocessor(self, example):
        return {"src_texts": "Scitail sentence1 : {} sentence2: {}".format(example['sentence1'], example['sentence2']),
                "tgt_texts": str(example['gold_label'])}


class MRPCTaskDataset(AbstractTaskDataset):
    task = Task(name="mrpc", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mrpc', split=split)

    def preprocessor(self, example):
        return {"src_texts": "MRPC sentence1 : {} sentence2: {}".format(example['sentence1'], example['sentence2']),
                "tgt_texts": str(example['label'])}


class COLATaskDataset(AbstractTaskDataset):
    task = Task(name="cola", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola', split=split)

    def preprocessor(self, example):
        return {"src_texts": "COLA sentence : {}".format(example['sentence']),
                "tgt_texts": str(example['label'])}


class SST2TaskDataset(AbstractTaskDataset):
    task = Task(name="sst2", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2', split=split)

    def preprocessor(self, example):
        return {"src_texts": "SST2 sentence : {}".format(example['sentence']),
                "tgt_texts": str(example['label'])}



class QQPTaskDataset(AbstractTaskDataset):
    task = Task(name="qqp", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp', split=split)

    def preprocessor(self, example):
        return {"src_texts": "QQP question1 : {} question2: {}".format(example['question1'], example['question2']),
                "tgt_texts": str(example['label'])}


"""
class STSBTaskDataset(AbstractTaskDataset):
    task = Task(name="stsb", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': 3}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb', split=split)

    def preprocessor(self, example):
        return {"src_texts": "STSB sentence1 : {} sentence2: {}".format(example['sentence1'], example['sentence2']),
                "tgt_texts": str(example['label'])}
"""


class MNLITaskDataset(AbstractTaskDataset):
    task = Task(name="mnli", category="classification")
    label_list = ["0", "1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train", "validation": "validation_mismatched", "test": "validation_matched"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split)

    def preprocessor(self, example):
        return {"src_texts": "MNLI premise : {} hypothesis : {}".format(example['premise'], example['hypothesis']),
                "tgt_texts": str(example['label'])}


class QNLITaskDataset(AbstractTaskDataset):
    task = Task(name="qnli", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split)

    def preprocessor(self, example):
        return {"src_texts": "QNLI question : {} sentence : {}".format(example['question'], example['sentence']),
                "tgt_texts": str(example['label'])}



class RTETaskDataset(AbstractTaskDataset):
    task = Task(name="rte", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte', split=split)

    def preprocessor(self, example):
        return {"src_texts": "RTE sentence1 : {} sentence2 : {}".format(example['sentence1'], example['sentence2']),
                "tgt_texts": str(example['label'])}



class WNLITaskDataset(AbstractTaskDataset):
    task = Task(name="wnli", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'wnli', split=split)

    def preprocessor(self, example):
        return {"src_texts": "WNLI sentence1 : {} sentence2 : {}".format(example['sentence1'], example['sentence2']),
                "tgt_texts": str(example['label'])}


TASK_MAPPING = OrderedDict([
     ('squad', SquadTaskDataset),
     ('imdb', IMDBTaskDataset),
     ('boolq', BoolQTaskDataset),
     ('snli', SNLITaskDataset),
     ('scitail', ScitailTaskDataset),
     ('mrpc', MRPCTaskDataset),
     ('trec', TRECTaskDataset),
     ('yelp_polarity', YelpPolarityTaskDataset),
     ('wmt16-ro-en', WMT16ROENTaskDataset),
     ('wmt14-hi-en', WMT14HIENTaskDataset),
     ('wmt16-en-ro', WMT16ENROTaskDataset),
     ('wmt16-ro-en', WMT16ROENTaskDataset),
     ('wmt16-en-cs', WMT16ENCSTaskDataset),
     ('iwslt2017-ro-nl', IWSLT2017RONL),
     ('iwslt2017-en-nl', IWSLT2017ENNL),
     ('cola', COLATaskDataset),
     ('sst2', SST2TaskDataset),
     ('qqp', QQPTaskDataset),
     ('mnli', MNLITaskDataset),
     ('qnli', QNLITaskDataset),
     ('rte', RTETaskDataset),
     ('wnli', WNLITaskDataset),
     ('wmt16-en-fi', WMT16ENFITaskDataset)
]
)


class AutoTask:
    @classmethod
    def get(self, task_name):
        if task_name in TASK_MAPPING:
            return TASK_MAPPING[task_name]()

        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )


class TaskCollator:
    def __init__(self, tokenizer, data_args, tpu_num_cores=None, return_targets=False, task=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        ######### this is for contrastive loss.
        # TODO: here we need to think how to define the labels for translation, ... tasks and
        # TODO: Also how we can get it to work with multiple datasets.
        self.return_targets = return_targets
        if self.return_targets:
            self.task = task
            TaskDataset = AutoTask.get(task)
            if TaskDataset.task.category != "classification":
                raise NotImplementedError("We can only return the targets for a classification task.")
            self.label_to_id = {v:i for i, v in enumerate(TaskDataset.label_list)}

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        # because of padding="longest" this does not work to be done in dataset part.
        batch = self._encode(batch)
        input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
        )
        decoder_input_ids = self._shift_right_t5(labels)
        output_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
        if self.return_targets:
            output_batch["targets"] = batch["targets"]
        return output_batch

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
        if self.return_targets:
            batch_encoding["targets"] = torch.tensor([self.label_to_id[x["tgt_texts"]] for x in batch])
        return batch_encoding.data


class TaskDataLoader:
    """Wrapper around dataloader to keep the task names."""
    def __init__(self, task, dataset, batch_size=8,
                 collate_fn=None, drop_last=False,
                 num_workers=0, sampler=None):
        self.dataset = dataset
        self.task = task
        self.batch_size = batch_size 
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      collate_fn=collate_fn,
                                      drop_last=drop_last,
                                      num_workers=num_workers)
    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task"] = self.task
            yield batch


# Note not to use itertools.cycle since it is
# doing some caching under the hood, resulting
# in issues in the dataloading pipeline.
# see https://docs.python.org/3.7/library/itertools.html?highlight=cycle#itertools.cycle
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class MultiTaskDataLoader:
    """Given a dictionary of task: dataset, returns a multi-task dataloader
    which uses temperature sampling to sample different datasets."""

    def __init__(self,  max_steps, tasks_to_datasets, batch_size=8, collate_fn=None,
                 drop_last=False, num_workers=0, temperature=100.0):
        # Computes a mapping from task to dataloaders.
        self.task_to_dataloaders = {}
        for task, dataset in tasks_to_datasets.items():
            dataloader = TaskDataLoader(task, dataset, batch_size,
                collate_fn=collate_fn, drop_last=drop_last, num_workers=num_workers)
            self.task_to_dataloaders.update({task: dataloader})
        self.tasknames = list(self.task_to_dataloaders.keys())

        # Computes the temperature sampling weights.
        self.sampling_weights = self.temperature_sampling(self.dataloader_sizes.values(), temperature)
        self.dataiters = {k: cycle(v) for k, v in self.task_to_dataloaders.items()}
        self.max_steps = max_steps

    def temperature_sampling(self, dataset_sizes, temp):
        total_size = sum(dataset_sizes)
        weights = np.array([(size / total_size) ** (1.0 / temp) for size in dataset_sizes])
        return weights/np.sum(weights)

    @property
    def dataloader_sizes(self):
        if not hasattr(self, '_dataloader_sizes'):
            self._dataloader_sizes = {k: len(v) for k, v in self.task_to_dataloaders.items()}
        return self._dataloader_sizes

    def __len__(self):
        return sum(v for k, v in self.dataloader_sizes.items())

    def num_examples(self):
        return sum(len(dataloader.dataset) for dataloader in self.task_to_dataloaders.values())

    def __iter__(self):
        for i in range(self.max_steps):
            taskname = np.random.choice(self.tasknames, p=self.sampling_weights)
            dataiter = self.dataiters[taskname]
            outputs = next(dataiter)
            yield outputs
