"""Implements a T5 trainer class doing training and evaluation."""

import collections
from typing import Any, Dict, Optional, Tuple, Union
import warnings
import os
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from seq2seq.samplers import MultiTaskBatchSampler
from torch.utils.data import DistributedSampler, RandomSampler
from transformers import PreTrainedModel, logging
from .trainer  import Trainer
from transformers.configuration_fsmt import FSMTConfig
from transformers.file_utils import is_torch_tpu_available
from torch.utils.data.dataloader import DataLoader

from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from transformers.trainer_pt_utils import reissue_pt_warnings

from seq2seq.utils import use_task_specific_params, reset_config

logger = logging.get_logger(__name__)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}
import numpy as np

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

from transformers.trainer_utils import HPSearchBackend
from seq2seq.utils import upload


class T5Trainer(Trainer):
    def __init__(self, config=None, data_args=None, dataset_sizes=None, task_to_compute_metrics=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if config is None:
            assert isinstance(
                self.model, PreTrainedModel
            ), f"If no `config` is passed the model to be trained has to be of type `PreTrainedModel`, but is {self.model.__class__}"
            self.config = self._actual_model(self.model).config
        else:
            self.config = config

        self.task_to_compute_metrics = task_to_compute_metrics
        self.dataset_sizes = dataset_sizes
        self.data_args = data_args
        self.vocab_size = self.config.tgt_vocab_size if isinstance(self.config, FSMTConfig) else self.config.vocab_size
        self.gcs_bucket=self.args.gcs_bucket

        if self.args.label_smoothing != 0 or (self.data_args is not None and self.data_args.ignore_pad_token_for_loss):
            assert (
                self.config.pad_token_id is not None
            ), "Make sure that `config.pad_token_id` is correcly defined when ignoring `pad_token` for loss calculation or doing label smoothing."

        if self.config.pad_token_id is None and self.config.eos_token_id is not None:
            logger.warn(
                f"The `config.pad_token_id` is `None`. Using `config.eos_token_id` = {self.config.eos_token_id} for padding.."
            )

        if self.args.label_smoothing == 0:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        else:
            # dynamically import label_smoothed_nll_loss
            from seq2seq.utils import label_smoothed_nll_loss

            self.loss_fn = label_smoothed_nll_loss


    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use
        something else, you can pass a tuple in the Trainer's init through
        :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.adafactor:
                self.optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    scale_parameter=False,
                    relative_step=False,
                )

            else:
                self.optimizer = AdamW(
                    optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon
                )

        if self.lr_scheduler is None:
            self.lr_scheduler = self._get_lr_scheduler(num_training_steps)
        else:  # ignoring --lr_scheduler
            logger.warn("scheduler is passed to `Seq2SeqTrainer`, `--lr_scheduler` arg is ignored.")

    def _get_lr_scheduler(self, num_training_steps):
        schedule_func = arg_to_scheduler[self.args.lr_scheduler]
        if self.args.lr_scheduler == "constant":
            scheduler = schedule_func(self.optimizer)
        elif self.args.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(self.optimizer, num_warmup_steps=self.args.warmup_steps)
        else:
            scheduler = schedule_func(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
        return scheduler

    """
    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset, self.data_groups)
        else:
            if self.args.sortish_sampler:
                self.train_dataset.make_sortish_sampler(
                    self.args.per_device_train_batch_size, 
                    distributed=(self.args.n_gpu > 1 and self.args.local_rank != -1)
                )

            return (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )
    """


    def get_eval_dataloader(self, eval_dataset: Dataset, task:str) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is
                an :obj:`datasets.Dataset`, columns not accepted by the ``model.forward()``
                method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            dataset=eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    # : Optional[Dataset] = None
    def evaluate(self) -> Dict[str, float]: #eval_datasets, task_to_compute_metrics) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """

        # TODO: make this condition for all.
        #if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
        #    raise ValueError("eval_dataset must implement __len__")
        tasks_metrics = {}
        model_config = self.model.config
        for eval_task, eval_dataset in self.eval_dataset.items():
            use_task_specific_params(self.model, eval_task)
            eval_dataloader = self.get_eval_dataloader(eval_dataset, eval_task)
            self.compute_metrics = self.task_to_compute_metrics[eval_task]
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None, # self.compute_metrics[eval_task]
            )
            # Prefix outputs with the tasks.
            tasks_metric = {eval_task+"_"+k: v for k, v in output.metrics.items()}
            tasks_metrics.update(tasks_metric)
            self.log(tasks_metric)

            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())
            reset_config(self.model, model_config)
        # TODO: this is not doing anything.
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        return tasks_metrics

    def _compute_loss(self, model, inputs, labels):
        if self.args.label_smoothing == 0:
            if self.data_args is not None and self.data_args.ignore_pad_token_for_loss:
                # force training to ignore pad token
                logits = model(**inputs, use_cache=False)[0]
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            else:
                # compute usual loss via models
                loss, logits = model(**inputs, labels=labels, use_cache=False)[:2]
        else:
            # compute label smoothed loss
            logits = model(**inputs, use_cache=False)[0]
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss, _ = self.loss_fn(lprobs, labels, self.args.label_smoothing, ignore_index=self.config.pad_token_id)
        return loss, logits


    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases (even distributed/parallel), self.model is always a reference
        # to the model we want to save.
        if hasattr(model, "module"):
            assert model.module is self.model, f"Module {model.module} should be a reference to self.model"
        else:
            assert model is self.model, f"Model {model} should be a reference to self.model"
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is not None and trial is not None:
            run_id = trial.number if self.hp_search_backend == HPSearchBackend.OPTUNA else tune.get_trial_id()
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            output_dir = os.path.join(self.args.output_dir, run_name, checkpoint_folder)
        else:
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

            self.store_flos()
        self.save_model(output_dir)

        # Save optimizer and scheduler
        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                reissue_pt_warnings(caught_warnings)
        elif self.is_world_process_zero():
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            reissue_pt_warnings(caught_warnings)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.is_world_process_zero():
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # Maybe delete some older checkpoints.
        if self.is_world_process_zero():
            self._rotate_checkpoints(use_mtime=True)

        if self.is_world_process_zero() and self.args.gcs_bucket is not None:
            # TODO: make sure enter this like this.
            #gcs_bucket="ruse-xcloud-bucket"
            upload(self.args.output_dir, self.args.gcs_bucket)

    def get_sharded_data(self, num_replicas, rank):
        """Returns the sharded data belonging to the given rank."""
        sharded_dataset_names_to_datasets = {}
        for dataset_name, dataset in self.train_dataset.items():
            sharded_data = dataset.shard(num_replicas, rank)
            sharded_dataset_names_to_datasets.update({dataset_name: sharded_data})
        self.train_dataset = sharded_dataset_names_to_datasets
        return self.train_dataset

    """
    def get_train_dataset_shards(self):
        #In case of multiprocessing, returns the sharded data for the given rank.
        if is_torch_tpu_available():
            if xm.xrt_world_size() > 1:
                return self.get_sharded_data(num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
            else:
                return self.train_dataset
        elif self.args.local_rank != -1:
                return self.get_sharded_data(num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        else:
            return self.train_dataset
    """

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        #train_dataset = self.get_train_dataset_shards()
        # TODO: we need to make sure that the number of batches are correctly computed
        #   and this is consistent across the cores.
        multitask_sampler = MultiTaskBatchSampler(self.dataset_sizes, self.args.train_batch_size,
                self.args.temperature)
        return DataLoader(self.train_dataset, batch_sampler=multitask_sampler,
                                collate_fn=self.data_collator)

    def compute_loss(self, model, inputs):
        for name, param in model.named_parameters():
            if param.requires_grad:
               print(name)
        labels = inputs.pop("labels")
        loss, _ = self._compute_loss(model, inputs, labels)
        return loss

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model.
                Most models expect the targets under the argument :obj:`labels`.
                Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        inputs = self._prepare_inputs(inputs)
        # TODO: we set these in evalute function, does this function is called alone too?
        # TODO: the arguments needs to be handled per task.
        gen_kwargs = {
            "max_length": self.config.max_length,
            #self.data_args.val_max_target_length
            #if self.data_args is not None
            #else self.config.max_length,
            "num_beams": self.config.num_beams #self.data_args.eval_beams if self.data_args is not None else self.config.num_beams,
        }
        gen_kwargs["task"] = inputs["task"]
        if self.args.predict_with_generate and not self.args.prediction_loss_only:
            generated_tokens = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        labels = inputs.pop("labels")
        with torch.no_grad():
            # compute loss on predict data
            loss, logits = self._compute_loss(model, inputs, labels)

        loss = loss.mean().detach()
        if self.args.prediction_loss_only:
            return (loss, None, None)

        logits = generated_tokens if self.args.predict_with_generate else logits

        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, logits, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                f"Make sure that either `config.pad_token_id` or `config.eos_token_id`"
                f" is defined if tensor has to be padded to `max_length`={max_length}"
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor



