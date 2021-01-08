# coding=utf-8
# Copyright 2010, The T5 Authors and HuggingFace Inc.
# Copyright 2020 Google LLC
# Modified from the original HuggingFace version.
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
import sys

import datasets
import json
import logging
import os
import numpy as np
from pathlib import Path
from third_party.models import T5Config, T5ForConditionalGeneration
from third_party.trainers import T5Trainer
from third_party.utils import (
    check_output_dir
)
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import EvaluationStrategy

from seq2seq.adapters import AdapterController, AutoAdapterConfig
from seq2seq.data import AutoTask
from seq2seq.third_party.utils import TaskCollator
from seq2seq.metrics import build_compute_metrics_fn
from seq2seq.training_args import Seq2SeqTrainingArguments, ModelArguments, DataTrainingArguments, \
    AdapterTrainingArguments
from seq2seq.utils import T5CheckpointCallback, freezing_params, get_last_checkpoint_path, create_dir, \
    handle_metrics, get_training_args

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py or by passing
    # the --help flag to this script. We now keep distinct sets of args, for a cleaner
    # separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdapterTrainingArguments))

    # For running on multiple gpus with torch.distributed.launch, it adds a local_rank paramter, to allow the parser
    # still use the config file, we add the local_rank to the config file.
    if len(sys.argv) == 3 and sys.argv[1].startswith("--local_rank") and sys.argv[2].endswith(".json"):
        args_dict = json.loads(Path(sys.argv[2]).read_text())
        args_dict.update({'local_rank': int(sys.argv[1].split('=')[-1])})
        model_args, data_args, training_args, adapter_args = parser.parse_dict(args_dict)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

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

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = T5Config.from_pretrained(
        model_args.config_name if model_args.config_name else \
            model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout",
                          "attention_dropout", "fixed_length_emb",
                          "encoder_projection", "encoder_pooling",
                          "projection_length", "only_projection_bottleneck",
                          "concat_projection_token", "train_adapters")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))

    # Gets the adapter config and updates the specified parameters.
    if training_args.train_adapters:
        adapter_config = AutoAdapterConfig.get(adapter_args.adapter_config_name)
        adapter_config.input_dim = config.d_model
        adapter_config.tasks = data_args.tasks
        extra_adapter_params = ("task_embedding_dir",
                                "task_embedding_dim",
                                "add_layer_norm_before_adapter",
                                "add_layer_norm_after_adapter",
                                "reduction_factor",
                                "hidden_dim",
                                "non_linearity",
                                "train_task_embeddings",
                                "projected_task_embedding_dim",
                                "add_adapters_in_decoder",
                                "add_adapter_in_feed_forward",
                                "add_adapter_in_self_attention",
                                "task_hidden_dim",
                                "conditional_layer_norm",
                                "one_layer_adapter_hyper_net",
                                "adapter_hyper_net_with_bias",
                                "one_layer_adapter_hyper_net_with_linear",
                                "parametric_task_embedding",
                                "conditional_layer_norm_for_T5",
                                "train_adapters_blocks",
                                "remove_original_layer_norms",
                                "unique_hyper_net",
                                "unique_hyper_net_layer_norm")
        for p in extra_adapter_params:
            if hasattr(adapter_args, p) and hasattr(adapter_config, p):
                setattr(adapter_config, p, getattr(adapter_args, p))
            else:
                logger.warning(f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute")
        adapter_config.device = training_args.device
    else:
        adapter_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else \
            model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    if model_args.not_load_t5_checkpoint:
        model = T5ForConditionalGeneration(config=config, adapter_config=adapter_config)
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            adapter_config=adapter_config
        )

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # freezing the parameters.
    if training_args.do_train:
        freezing_params(model, training_args, model_args, adapter_args)

    if training_args.print_num_parameters:
        logger.info(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info("Parameter name %s", name)
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Total trainable parameters %s", total_trainable_params)
    # Gets the training/test/validation datasets.
    dataset_class = AutoTask
    if training_args.do_train:
        train_datasets = [dataset_class.get(task, seed=data_args.data_seed).get_dataset(
            split="train", n_obs=data_args.n_train, add_prefix=False if training_args.train_adapters else True)
            for task in data_args.tasks]
        dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
        train_dataset = datasets.concatenate_datasets(train_datasets)
    training_args.remove_unused_columns = False
    eval_datasets = ({task: dataset_class.get(task, seed=data_args.data_seed).get_dataset(
        split="validation", n_obs=data_args.n_val,
        add_prefix=False if training_args.train_adapters else True,
        split_validation_test=training_args.split_validation_test)
                         for task in data_args.eval_tasks}
                     if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
                     else None
                     )
    test_dataset = (
        {task: dataset_class.get(task, seed=data_args.data_seed).get_dataset(
            split="test", n_obs=data_args.n_test,
            add_prefix=False if training_args.train_adapters else True,
            split_validation_test=training_args.split_validation_test)
            for task in data_args.eval_tasks} if training_args.do_test else None
    )
    # Defines the metrics for evaluation.
    compute_metrics_fn = (
        build_compute_metrics_fn(data_args.eval_tasks, tokenizer) if training_args.predict_with_generate else None
    )
    # Defines the trainer.
    trainer = T5Trainer(
        model=model,
        config=config,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_datasets,
        data_collator=TaskCollator(tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores),
        compute_metrics=None,
        multi_task_compute_metrics=compute_metrics_fn,
        data_args=data_args,
        dataset_sizes=dataset_sizes if training_args.do_train else None,
        callbacks=[T5CheckpointCallback()],
        adapter_config=adapter_config
    )
    if trainer.is_world_process_zero():
        arguments = get_training_args([model_args, data_args, training_args, adapter_args])
        handle_metrics("arguments", arguments, training_args.output_dir, training_args.gcs_bucket)

    # Trains the model.
    if training_args.do_train:
        trainer.train(
            model_path=get_last_checkpoint_path(training_args.output_dir) \
                if (os.path.isdir(training_args.output_dir) and not training_args.optimize_from_scratch) else None,
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
            tokenizer.save_pretrained(training_args.output_dir)
            if training_args.save_task_embeddings:
                for task, task_embedding in model.task_embedding_controller.task_to_embeddings.items():
                    create_dir(training_args.save_task_embeddings_dir)
                    np.save(os.path.join(training_args.save_task_embeddings_dir,
                                         '{}.npy'.format(task)), task_embedding.data.detach().cpu().numpy())

    # Evaluation
    all_metrics = {}
    if training_args.do_eval or training_args.do_test:
        if trainer.is_world_process_zero():
            # By default we load  the model from last checkpoint path,
            # in case of saving the model with the best metrics, make sure to
            # set save_total = 1 so the best model is loaded here.
            # if not exists returns the path to the output_dir.
            last_checkpoint_path = get_last_checkpoint_path(training_args.output_dir)
            config = T5Config.from_pretrained(
                last_checkpoint_path,
                cache_dir=model_args.cache_dir)
            model = T5ForConditionalGeneration.from_pretrained(
                last_checkpoint_path,
                from_tf=".ckpt" in training_args.output_dir,
                config=config,
                cache_dir=model_args.cache_dir,
                adapter_config=adapter_config
            )
            # NOTE: if trainer is not re-defined, there is a bug in the codes, that making
            # huggingface codes does not using the best checkpoint.
            trainer = T5Trainer(
                model=model,
                config=config,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_datasets,
                data_collator=TaskCollator(tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores),
                compute_metrics=None,
                multi_task_compute_metrics=compute_metrics_fn,
                data_args=data_args,
                dataset_sizes=dataset_sizes if training_args.do_train else None,
                callbacks=[T5CheckpointCallback()],
                adapter_config=adapter_config
            )

        if training_args.train_adapters:
            if adapter_args.adapter_config_name == "adapter" and data_args.adapters is not None:
                for name, sub_module in model.named_modules():
                    task_to_adapter = {eval_task: adapter for eval_task, adapter in
                                       zip(data_args.eval_tasks, data_args.adapters)}
                    if isinstance(sub_module, AdapterController):
                        sub_module.set_task_to_adapter_map(task_to_adapter)
            if adapter_args.adapter_config_name in ["meta-adapter"]:
                # If this is parametric, then the evaluation task should be part of tasks
                # and the embeddings needs to be trained.
                if not adapter_args.parametric_task_embedding:
                    model.task_embedding_controller.set_task_embeddings(eval_datasets.keys(),
                                                                        parametric=adapter_args.parametric_task_embedding)

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="val")
        if trainer.is_world_process_zero():
            handle_metrics("val", metrics, training_args.output_dir, training_args.gcs_bucket)
            all_metrics.update(metrics)

    if training_args.do_test:
        metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        if trainer.is_world_process_zero():
            handle_metrics("test", metrics, training_args.output_dir, training_args.gcs_bucket)
            all_metrics.update(metrics)

    return all_metrics


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
