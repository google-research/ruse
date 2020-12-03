import logging
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments

from seq2seq.models import PROJECTION_MAPPING
from seq2seq.models import POOLING_MAPPING

from transformers.optimization import (
    Adafactor,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}

logger = logging.getLogger(__name__)


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Parameters:
        label_smoothing (:obj:`float`, `optional`, defaults to 0):
            The label smoothing epsilon to apply (if not zero).
        sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to SortishSamler or not. It sorts the inputs according to lenghts in-order to minimizing the padding size.
        predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """

    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear",
        metadata={"help": f"Which lr scheduler to use. Selected in {sorted(arg_to_scheduler.keys())}"},
    )
    fixed_length_emb: bool = field(default=None, metadata={"help": f"Whether to learn fixed length embeddings or not."
        f"Goes into model.config."})
    encoder_projection: Optional[str] = field(
        default=None,
        metadata={"help": f"Projection layer to use in case of learning fixed length embeddings. "
        f"Selected in {sorted(PROJECTION_MAPPING.keys())}. Goes into model.config."}
    )
    encoder_pooling: Optional[str] = field(
        default=None,
        metadata={"help": f"Pooling layer to use in case of learning fixed length embeddings. "
                  f"Selected in {sorted(POOLING_MAPPING.keys())}. Goes into model.config."}
    )
    projection_length: Optional[int] = field(
        default=None,
        metadata={"help": f"Size of projection layer in case of learning fixed length embeddings. "
                  f"Goes into model.config."}
    )
    only_projection_bottleneck: bool = field(default=False, metadata={"help": f"If set, only passes the"
    f" projection bottleneck to the decoder."})
    concat_projection_token: bool = field(default=False, metadata={"help": f"If set, it concatenates the"
    f"computed fixed vector to the embeddings."})
    gcs_bucket: Optional[str] = field(
      default=None,
      metadata={"help":f"gs bucket."}
    )
    train_adapters: bool = field(default=False, metadata={"help":
        "Train an adapter instead of the full model."})
    temperature: int = field(default=1, metadata={"help": "Defines the temperature"
        "value for sampling across the multiple datasets."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    not_load_t5_checkpoint: bool = field(default=False, metadata={"help": "whether to load the checkpoint."})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    sampling: bool = field(default=False, metadata={"help": "Whether to sample training datasets."})
    tasks: Optional[List[str]] = field(
        default="MRPC",
        metadata={"help": "Task name from the list of registered tasks."},
    )
    eval_tasks: Optional[List[str]] = field(
        default="MRPC",
        metadata={"help": "Evaluation task name from the list of registered tasks."},
    )
    adapters: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from adapters to the tasks."}
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
    val_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=-1, metadata={"help": "# test examples. -1 means use all."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
