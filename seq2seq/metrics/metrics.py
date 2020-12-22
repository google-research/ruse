from collections import OrderedDict

import numpy as np
from logging import getLogger
from third_party.utils import calculate_rouge, calculate_bleu, lmap
from transformers import EvalPrediction, PreTrainedTokenizer
from typing import Callable, Dict, List, Tuple

from seq2seq.data import TASK_MAPPING

logger = getLogger(__name__)


def accuracy(output_lns, refs_lns) -> dict:
    """Computes the avarage accuracy."""
    return {"acc": (np.array(output_lns) == np.array(refs_lns)).mean()}





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
    return task_to_compute_metrics
