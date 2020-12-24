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
"""Implements different tasks."""
from collections import OrderedDict

import abc
import datasets
import functools
import numpy as np
from typing import Callable, Dict, Mapping, List

from seq2seq.metrics import metrics
from .utils import round_stsb_target, compute_task_max_decoding_length


class AbstractTaskDataset(abc.ABC):
    name = NotImplemented
    task_specific_config: Dict = NotImplemented
    preprocessor: Callable = NotImplemented
    metrics: List[Callable] = NotImplemented
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}

    def get_sampled_split(self, split, n_obs=None):
        split = self.split_to_data_split[split]
        ###############################
        # this is to check.
        dataset = self.load_dataset(split=split)
        if len(dataset) < n_obs:
            n_obs = len(dataset)
        ###############################
        if n_obs is not None:
            split = split + "[:{}]".format(n_obs)
        return split

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split, script_version="master")

    def get_dataset(self, split, n_obs=None, add_prefix=True):
        split = self.get_sampled_split(split, n_obs)
        dataset = self.load_dataset(split=split)
        dataset = dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                              remove_columns=dataset.column_names)
        return dataset

    def seq2seq_format(self, src_strs, tgt_strs, add_prefix=False, prefix=None):
        src_prefix = self.name if prefix is None else prefix
        src_strs = [src_prefix] + src_strs if add_prefix else src_strs
        return {"src_texts": ' '.join(src_strs),
                "tgt_texts": ' '.join(tgt_strs),
                "task": self.name}


class SquadTaskDataset(AbstractTaskDataset):
    name = "squad"
    split_to_data_split = {"train": "train", "validation": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"], "context:", example["context"]]
        tgt_texts = [example["answers"]["text"][0]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class IMDBTaskDataset(AbstractTaskDataset):
    name = "imdb"
    split_to_data_split = {"train": "train", "validation": "test"}
    label_list = ["pos", "neg"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["text"]]
        tgt_texts = [example["label"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class BoolQTaskDataset(AbstractTaskDataset):
    name = "boolq"
    label_list = ["False", "True"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metrics = [metrics.accuracy]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"], "passage:", example["passage"]]
        tgt_texts = [str(example["answer"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SNLITaskDataset(AbstractTaskDataset):
    name = "snli"
    label_list = ["0", "1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metrics = [metrics.accuracy]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"], "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class IWSLT2017RONL(AbstractTaskDataset):
    name = "iwslt2017-ro-nl"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-nl"

    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", 'iwslt2017-ro-nl', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["ro"]]
        tgt_texts = [example['translation']["nl"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate Romanian to Dutch")


class IWSLT2017ENNL(AbstractTaskDataset):
    name = "iwslt2017-en-nl"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"en-nl"

    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", 'iwslt2017-en-nl', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["nl"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Dutch")


class WMT16ENROTaskDataset(AbstractTaskDataset):
    name = "wmt16-en-ro"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["ro"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Romanian")


class WMT16ROENTaskDataset(AbstractTaskDataset):
    name = "wmt16-ro-en"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["ro"]]
        tgt_texts = [example['translation']["en"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate Romanian to English")


class WMT16ENCSTaskDataset(AbstractTaskDataset):
    name = "wmt16-en-cs"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"cs-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["cs"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Czech")


class WMT16ENFITaskDataset(AbstractTaskDataset):
    name = "wmt16-en-fi"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"fi-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["fi"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Finnish")


class WMT14HIENTaskDataset(AbstractTaskDataset):
    name = "wmt14-hi-en"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"hi-en"

    def load_dataset(self, split):
        return datasets.load_dataset("wmt14", self.pair, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["hi"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Hindi")


class TRECTaskDataset(AbstractTaskDataset):
    name = "trec"
    label_list = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset("trec", split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [example['label-coarse']]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class YelpPolarityTaskDataset(AbstractTaskDataset):
    name = "yelp_polarity"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train", "validation": "test", "test": "test"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset("yelp_polarity", split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class ScitailTaskDataset(AbstractTaskDataset):
    name = "scitail"
    label_list = ["entailment", "neutral"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        dataset = datasets.load_dataset("scitail", "snli_format", split=split)
        return dataset

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'], "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['gold_label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MRPCTaskDataset(AbstractTaskDataset):
    name = "mrpc"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.f1_score_with_invalid, metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mrpc', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'], "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class COLATaskDataset(AbstractTaskDataset):
    name = "cola"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.matthews_corrcoef]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SST2TaskDataset(AbstractTaskDataset):
    name = "sst2"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class STSBTaskDataset(AbstractTaskDataset):
    name = "stsb"
    label_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'], "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QQPTaskDataset(AbstractTaskDataset):
    name = "qqp"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.f1_score_with_invalid, metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question1:", example['question1'], "question2:", example["question2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MNLITaskDataset(AbstractTaskDataset):
    name = "mnli"
    label_list = ["0", "1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train", "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'], "hypothesis", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QNLITaskDataset(AbstractTaskDataset):
    name = "qnli"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['question'], "sentence:", example["sentence"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class RTETaskDataset(AbstractTaskDataset):
    name = "rte"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'], "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WNLITaskDataset(AbstractTaskDataset):
    name = "wnli"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'wnli', split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'], "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SocialIQaTaskDataset(AbstractTaskDataset):
    name = "socialiqa"
    label_list = ["1", "2", "3"]


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
    ('stsb', STSBTaskDataset),
    ('qqp', QQPTaskDataset),
    ('mnli', MNLITaskDataset),
    ('qnli', QNLITaskDataset),
    ('rte', RTETaskDataset),
    ('wnli', WNLITaskDataset),
    ('wmt16-en-fi', WMT16ENFITaskDataset)]
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
