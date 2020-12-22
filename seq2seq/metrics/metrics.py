# Copyright 2020 The T5 Authors.
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

from collections import OrderedDict

import numpy as np
import scipy
from logging import getLogger
import sklearn
from third_party.utils import calculate_rouge, calculate_bleu, lmap
from transformers import EvalPrediction, PreTrainedTokenizer
from typing import Callable, Dict, List, Tuple
import functools



logger = getLogger(__name__)


def accuracy(output_lns, refs_lns) -> dict:
    """Computes the average accuracy."""
    return {"acc": (np.array(output_lns) == np.array(refs_lns)).mean()}

def pearson_corrcoef(targets, predictions)-> dict:
  """Computes Pearson correlation coefficient."""
  return {"pearson_corrcoef":
              100 * scipy.stats.pearsonr(targets, predictions)[0]}

def spearman_corrcoef(targets, predictions)-> dict:
  """Computes Spearman correlation coefficient."""
  return {"spearman_corrcoef":
              100 * scipy.stats.spearmanr(targets, predictions)[0]}


# This is from T5 paper.
def f1_score_with_invalid(targets, predictions)-> dict:
  """Compute F1 score, but any prediction != 0 or 1 is counted as incorrect.
  Args:
    targets: np.ndarray of targets, either 0 or 1
    predictions: np.ndarray of predictions, any integer value
  Returns:
    F1 score, where any prediction != 0 or 1 is counted as wrong.
  """
  targets, predictions = np.asarray(targets), np.asarray(predictions)
  # Get indices of invalid predictions
  invalid_idx_mask = np.logical_and(predictions != 0, predictions != 1)
  # For any prediction != 0 or 1, set it to the opposite of what the target is.
  predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
  return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions)}


def matthews_corrcoef(targets, predictions)-> dict:
    """Computes the Matthews correlation coefficient."""
    return {"mcc": 100 * sklearn.metrics.matthews_corrcoef(targets, predictions)}


def build_compute_metrics_fn(task_names: List[str],
                             tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:
    """Builds a dictionary from each task to the task metric."""

    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        return pred_str, label_str

    def summarization_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        rouge.update({"gen_len": summ_len})
        return rouge

    def translation_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        bleu: Dict = calculate_bleu(pred_str, label_str)
        gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        bleu.update({"gen_len": gen_len})
        return bleu

    def classification_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        acc: Dict = accuracy(pred_str, label_str)
        return acc


    def t5_wrapper_metrics(pred: EvalPrediction, metrics) -> Dict:
        pred_str, label_str = decode_pred(pred)
        eval_results = {}
        for metric in metrics:
            eval_results.update(metric(pred_str, label_str))
        return eval_results


    def tasks_metrics(task) -> Dict:
        from data.tasks import TASK_MAPPING
        return functools.partial(t5_wrapper_metrics, metrics=TASK_MAPPING[task].metrics) #compute_metrics_fn

    """
    def tasks_metrics(task=None) -> Dict:
        category = TASK_MAPPING[task].task.category
        compute_metrics_fn = CATEGORY_EVALUATION_MAPPING[category]
        logger.info(f"selected metric {compute_metrics_fn} for task {task}")
        return compute_metrics_fn
    
    CATEGORY_EVALUATION_MAPPING = OrderedDict(
        [('summarization', summarization_metrics),
         ('translation', translation_metrics),
         ('classification', classification_metrics)
         ]
    )
    task_to_compute_metrics = {task: tasks_metrics(task) for task in task_names}
    """

    task_to_compute_metrics = {task: tasks_metrics(task) for task in task_names}
    return task_to_compute_metrics

