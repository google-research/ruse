import numpy as np
from collections import OrderedDict
from typing import Callable, Dict, Mapping, List
import torch
from dataclasses import dataclass
import datasets
import abc
from transformers import T5Tokenizer
from torch.utils.data.dataloader import DataLoader
import functools


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
        {"train": "train", "validation": "validation", "test": "test"}

    def get_sampled_split(self, split, n_obs=None):
        split = self.split_to_data_split[split]
        if n_obs is not None:
            split = split + "[:{}]".format(n_obs)
        return split

    def add_prefix(self, text, prefix, add_prefix):
        """If add_prefix is set to true adds the prefix to the given text."""
        return prefix+" : "+text if add_prefix else text

    def load_dataset(self, split):
        return datasets.load_dataset(self.task.name, split=split)

    def get_dataset(self, split, n_obs=None, add_prefix=True):
        split = self.get_sampled_split(split, n_obs)
        dataset = self.load_dataset(split=split)
        dataset = dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                              remove_columns=dataset.column_names)
        return dataset


class SquadTaskDataset(AbstractTaskDataset):
    task = Task(name="squad", category="question_answering")
    split_to_data_split = {"train": "train", "validation": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = "question: {0} context: {1} ".format(example["question"], example["context"])
        return {"src_texts": self.add_prefix(src_texts, "SQUAD", add_prefix),
                "tgt_texts": example["answers"]["text"][0], "task": self.task.name}


class IMDBTaskDataset(AbstractTaskDataset):
    task = Task(name="imdb", category="classification")
    split_to_data_split = {"train": "train", "validation": "test"}
    label_list = ["pos", "neg"]
    task_specific_config = {'max_length': 3}

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix(example["text"], "imdb", add_prefix),
                "tgt_texts": str(example["label"]), "task": self.task.name}


class BoolQTaskDataset(AbstractTaskDataset):
    task_specific_config = {'max_length': 3}
    task = Task(name="boolq", category="classification")
    split_to_data_split = {"train": "train", "validation": "validation", "test":"validation"}
    label_list=["0", "1"]

    def preprocessor(self, example, add_prefix=True):
        src_texts = "question: {} passage: {}: ".format(example["question"], example["passage"])
        return {"src_texts": self.add_prefix(src_texts, "Boolq", add_prefix),
                "tgt_texts": str(example["answer"]), "task": self.task.name}


class SNLITaskDataset(AbstractTaskDataset):
    task_specific_config = {'max_length': 5, 'num_beams': 4}
    task = Task(name="snli", category="classification")
    task_specific_config = {'max_length': 3}
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}
    label_list = ["0", "1", "2"]

    def preprocessor(self, example, add_prefix=True):
        src_texts = "premise: {} hypothesis: {}".format(example["premise"], example["hypothesis"])
        return {"src_texts": self.add_prefix(src_texts, "SNLI", add_prefix),
                "tgt_texts": str(example["label"]), "task": self.task.name}


class MNLITaskDataset(AbstractTaskDataset):
    task_specific_config = {'max_length': 5, 'num_beams': 4}
    task = Task(name="mnli", category="classification")
    task_specific_config = {'max_length': 3}
    split_to_data_split = {"train": "train", "validation": "validation_mismatched",
                           "test": "validation_matched"}
    label_list = ["0", "1", "2"]

    def preprocessor(self, example, add_prefix=True):
        src_texts="premise: {} hypothesis: {}".format(example["premise"], example["hypothesis"])
        return {"src_texts": self.add_prefix(src_texts, "MNLI", add_prefix),
                "tgt_texts": str(example["label"]), "task": self.task.name}


# TODO: the class should get the name of pairs as a argument.
# and register each class with its own arguments.
class IWSLT2017RONL(AbstractTaskDataset):
    task = Task(name="iwslt2017-ro-nl", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-nl"

    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", 'iwslt2017-ro-nl', split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix(example['translation']["ro"],
                "Translate Romanian to Dutch", add_prefix),
                "tgt_texts": str(example['translation']["nl"]), "task": self.task.name}



class IWSLT2017ENNL(AbstractTaskDataset):
    task = Task(name="iwslt2017-en-nl", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"en-nl"

    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", 'iwslt2017-en-ko', split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix(example['translation']["en"],
                "Translate English to Dutch", add_prefix),
                "tgt_texts": str(example['translation']["nl"]), "task": self.task.name}



class WMT16ENROTaskDataset(AbstractTaskDataset):
    task = Task(name="wmt16-en-ro", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix(example['translation']["en"],
                                             "Translate English to Romanian",
                                             add_prefix),
                "tgt_texts": str(example['translation']["ro"]), "task": self.task.name}


class WMT16ROENTaskDataset(AbstractTaskDataset):
    task = Task(name="wmt16-ro-en", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": "{}{}".format(
                "Translate Romanian to English: " if add_prefix else "",
                example['translation']["ro"]),
                "tgt_texts": str(example['translation']["en"]), "task": self.task.name}


class WMT16ENCSTaskDataset(AbstractTaskDataset):
    task = Task(name="wmt16-en-cs", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"cs-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": "{} {}".format(
                "Translate English to Czech: " if add_prefix else "",
                example['translation']["en"]),
                "tgt_texts": str(example['translation']["cs"]), "task": self.task.name}


class WMT16ENFITaskDataset(AbstractTaskDataset):
    task = Task(name="wmt16-en-fi", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"fi-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example):
        return {"src_texts": "Translate English to Finnish:  {}".format(example['translation']["en"]),
                "tgt_texts": str(example['translation']["fi"]), "task": self.task.name}


class WMT14HIENTaskDataset(AbstractTaskDataset):
    task = Task(name="wmt14-hi-en", category="translation")
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"hi-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt14", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix(example['translation']["hi"],
                                             "Translate English to Romanian",
                                             add_prefix),
                "tgt_texts": str(example['translation']["en"]), "task": self.task.name}


class TRECTaskDataset(AbstractTaskDataset):
    task = Task(name="trec", category="classification")
    label_list=["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset("trec",  split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix("sentence :  {}".format(example['text']), "Trec", add_prefix),
                "tgt_texts": str(example['label-coarse']), "task": self.task.name}


class YelpPolarityTaskDataset(AbstractTaskDataset):
    task = Task(name="yelp_polarity", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train", "validation": "test", "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset("yelp_polarity",  split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix("sentence :  {}".format(example['text']), "Yelp Polarity", add_prefix),
                "tgt_texts": str(example['label']), "task": self.task.name}


class ScitailTaskDataset(AbstractTaskDataset):
    task = Task(name="scitail", category="classification")
    label_list = ["entailment", "neutral"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        dataset = datasets.load_dataset("scitail", "snli_format", split=split)
        return dataset

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix("sentence1: {} sentence2: {}".format(example['sentence1'], example['sentence2']),
                                             "Scitail", add_prefix),
                "tgt_texts": str(example['gold_label']), "task": self.task.name}


class MRPCTaskDataset(AbstractTaskDataset):
    task = Task(name="mrpc", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mrpc', split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix("sentence1 : {} sentence2: {}".format(example['sentence1'], example['sentence2']),
                "MRPC", add_prefix),
                "tgt_texts": str(example['label']), "task": self.task.name}


class COLATaskDataset(AbstractTaskDataset):
    task = Task(name="cola", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola', split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix("sentence : {}".format(example['sentence']), "COLA", add_prefix),
                "tgt_texts": str(example['label']), "task": self.task.name}


class SST2TaskDataset(AbstractTaskDataset):
    task = Task(name="sst2", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2', split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix("sentence : {}".format(example['sentence']), "SST2", add_prefix),
                "tgt_texts": str(example['label']), "task": self.task.name}



class QQPTaskDataset(AbstractTaskDataset):
    task = Task(name="qqp", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp', split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix("question1 : {} question2: {}".format(example['question1'], example['question2']),
                "QQP", add_prefix),
                "tgt_texts": str(example['label']), "task": self.task.name}


class MNLITaskDataset(AbstractTaskDataset):
    task = Task(name="mnli", category="classification")
    label_list = ["0", "1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train", "validation": "validation_mismatched", "test": "validation_matched"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts":
                self.add_prefix("premise : {} hypothesis : {}".format(example['premise'], example['hypothesis']), "MNLI", add_prefix),
                "tgt_texts": str(example['label']), "task": self.task.name}


class QNLITaskDataset(AbstractTaskDataset):
    task = Task(name="qnli", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix(
                "question: {} sentence: {}".format(example['question'], example['sentence']),
                "QNLI", add_prefix),
                "tgt_texts": str(example['label']), "task": self.task.name}



class RTETaskDataset(AbstractTaskDataset):
    task = Task(name="rte", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte', split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts": self.add_prefix(
                "sentence1 : {} sentence2 : {}".format(example['sentence1'], example['sentence2']),
                "RTE",
                add_prefix),
                "tgt_texts": str(example['label']), "task": self.task.name}



class WNLITaskDataset(AbstractTaskDataset):
    task = Task(name="wnli", category="classification")
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'wnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        return {"src_texts":
                self.add_prefix("sentence1 : {} sentence2 : {}".format(example['sentence1'], example['sentence2']),
                                "WNLI", add_prefix),
                "tgt_texts": str(example['label']), "task": self.task.name}


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
     ('wmt16-en-fi', WMT16ENFITaskDataset),
     ('mnli', MNLITaskDataset)
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
    def __init__(self, tokenizer, data_args, tpu_num_cores=None, return_targets=False, task=None): #, tasks=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        #self.task_to_id = {v: i for i, v in enumerate(tasks)}
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
        output_batch["task"] = batch["task"]
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
        tasks = [x["task"] for x in batch]
        # There should be only one task per batch.
        assert (len(set(tasks)) == 1)
        batch_encoding["task"] = tasks[0] #self.task_to_id[tasks[0]]
        return batch_encoding.data
