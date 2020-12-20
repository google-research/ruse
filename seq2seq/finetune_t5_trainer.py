import copy
import logging
import os
import sys

import torch.nn as nn
import datasets
from third_party.utils import build_compute_metrics_fn
from third_party.models import T5Config, T5ForConditionalGeneration
from third_party.trainers import T5Trainer
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_utils import EvaluationStrategy

from transformers.modeling_t5 import T5LayerNorm
from seq2seq.adapters import AdapterController, MetaAdapterController, AutoAdapterConfig
from seq2seq.data import AutoTask, TaskCollator
from seq2seq.training_args import Seq2SeqTrainingArguments, ModelArguments, DataTrainingArguments, \
  AdapterTrainingArguments
from third_party.utils import (
  assert_all_frozen,
  check_output_dir,
  freeze_embeds,
  freeze_params,
  lmap,
  save_json,
  write_txt_file,
)
from seq2seq.utils import T5CheckpointCallback
from seq2seq.utils import upload, use_task_specific_params, reset_config

logger = logging.getLogger(__name__)

if is_torch_tpu_available():
  import torch_xla.core.xla_model as xm


def shard_data(datasets, num_replicas, rank):
  """Returns the sharded data belonging to the given rank."""
  for i, dataset in enumerate(datasets):
    # shuffle needs to be per epoch as well.
    dataset = dataset.shuffle()
    sharded_dataset = dataset.shard(num_replicas, rank)
    datasets[i] = sharded_dataset
  return datasets


def freezing_params(model, training_args, model_args):
  if training_args.train_adapters:
    # Sets the last layer of decoder to be trained.
    freeze_params(model)
    for name, sub_module in model.named_modules():
      if isinstance(sub_module, (MetaAdapterController, AdapterController)):
        for param_name, param in sub_module.named_parameters():
          param.requires_grad = True
    for param in model.task_embedding_controller.parameters():
      param.requires_grad = True

  elif model_args.freeze_model_but_lm_head:
    freeze_params(model)
    for param in model.lm_head.parameters():
      param.requires_grad = True
  else:
    if model_args.freeze_embeds:
      freeze_embeds(model)
    if model_args.freeze_encoder:
      freeze_params(model.get_encoder())
      assert_all_frozen(model.get_encoder())

  if model_args.freeze_model_but_task_embeddings:
    freeze_params(model)
    for param in model.task_embedding_controller.parameters():
      param.requires_grad = True
  
  if model_args.unfreeze_lm_head:
    for param in model.lm_head.parameters():
      param.requires_grad = True

  if model_args.unfreeze_layer_norms:
    for name, sub_module in model.named_modules():
      if isinstance(sub_module, T5LayerNorm):
        for param_name, param in sub_module.named_parameters():
          param.requires_grad = True

def main():
  # See all possible arguments in src/transformers/training_args.py or by passing
  # the --help flag to this script. We now keep distinct sets of args, for a cleaner
  # separation of concerns.
  parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdapterTrainingArguments))

  if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
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
  # TODO: we need a better way to handle meta_adapters, and train_adapters
  # and task_embeddings_dir.
  extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout",
                        "attention_dropout", "fixed_length_emb",
                        "encoder_projection", "encoder_pooling",
                        "projection_length", "only_projection_bottleneck",
                        "concat_projection_token", "train_adapters")
  for p in extra_model_params:
    # TODO(rabeeh): this is a bug, if you set something to false, it wont be called.
    if getattr(training_args, p, None):
      assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
      setattr(config, p, getattr(training_args, p))

  # Gets the adapter config and updates the specified parameters.
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
                          "add_adapters_in_decoder")
  for p in extra_adapter_params:
    if hasattr(adapter_args, p):
        assert hasattr(adapter_config, p), f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute"
        setattr(adapter_config, p, getattr(adapter_args, p))
  adapter_config.device = training_args.device

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
    freezing_params(model, training_args, model_args)


  dataset_class = AutoTask
  if training_args.do_train:
    train_datasets = [dataset_class.get(task).get_dataset(
      split="train", n_obs=data_args.n_train, add_prefix=False if training_args.train_adapters else True)
      for task in data_args.tasks]
    # Shard the data if needed.
    # TODO: also add for distribued GPU training.
    # TODO: here we need to make sure shards are the same length across the cores.
    if is_torch_tpu_available() and xm.xrt_world_size() > 1:
      train_datasets = shard_data(train_datasets, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
    dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
    train_dataset = datasets.concatenate_datasets(train_datasets)
  # TODO: you should not do this, introduces bug.
  # train_dataset.set_format(type="torch", columns=['src_texts', 'tgt_texts'])
  training_args.remove_unused_columns = False

  # TODO: split varies.
  eval_datasets = ({task: dataset_class.get(task).get_dataset(
    split="validation", n_obs=data_args.n_val, add_prefix=False if training_args.train_adapters else True)
                     for task in data_args.eval_tasks}
                   if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
                   else None
                   )
  test_dataset = (
    {task: dataset_class.get(task).get_dataset(
      split="test", n_obs=data_args.n_test, add_prefix=False if training_args.train_adapters else True)
      for task in data_args.task}
    if training_args.do_predict
    else None
  )

  # TODO: this needs to get fixed, for now we do not need it.
  # Initialize our Trainer
  compute_metrics_fn = (
    build_compute_metrics_fn(data_args.eval_tasks, tokenizer) if training_args.predict_with_generate else None
  )

  # TODO: how does it get between different max_lengths?
  trainer = T5Trainer(
    model=model,
    config=config,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_datasets,
    data_collator=TaskCollator(tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores),
    compute_metrics=compute_metrics_fn,
    data_args=data_args,
    dataset_sizes=dataset_sizes if training_args.do_train else None,
    callbacks=[T5CheckpointCallback()]
  )
  # Training
  if training_args.do_train:
    trainer.train(
      model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
      trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
      tokenizer.save_pretrained(training_args.output_dir)

  # Evaluation
  eval_results = {}
  if training_args.do_eval:
    result = {}
    for eval_task, eval_dataset in eval_datasets.items():
      config = T5Config.from_pretrained(
        training_args.output_dir,  # "t5-base" for the baseline.
        cache_dir=model_args.cache_dir)
      model = T5ForConditionalGeneration.from_pretrained(
        training_args.output_dir,  # "t5-base" for the baseline.
        from_tf=".ckpt" in training_args.output_dir,
        config=config,
        cache_dir=model_args.cache_dir,
        adapter_config=adapter_config
      )
      if training_args.train_adapters:
        if adapter_args.adapter_config_name == "adapter" and data_args.adapters is not None:
          for name, sub_module in model.named_modules():
            task_to_adapter = {eval_task: adapter for eval_task, adapter in
                               zip(data_args.eval_tasks, data_args.adapters)}
            if isinstance(sub_module, AdapterController):
              sub_module.set_task_to_adapter_map(task_to_adapter)
        if adapter_args.adapter_config_name in ["meta-adapter", "parametric-meta-adapter"]:
          model.task_embedding_controller.update_task_embeddings([eval_task],
                    parametric=training_args.parametric_task_embedding)

      # if training_args.eval_output_dir is not None:
      #    training_args.output_dir = training_args.eval_output_dir
      model_config = model.config
      if training_args.do_finetune:
        train_dataset = dataset_class.get(eval_task).get_dataset(
          split="train", n_obs=data_args.n_finetune,
          add_prefix=False if training_args.train_adapters else True)
        dataset_sizes = [len(train_dataset)]
        compute_metrics_fn = (
          build_compute_metrics_fn(data_args.eval_tasks, tokenizer)
          if training_args.predict_with_generate else None
        )
        eval_training_args = copy.deepcopy(training_args)
        # sets the output_dir.
        eval_output_dir = training_args.eval_output_dir if training_args.eval_output_dir is not None else training_args.output_dir
        eval_training_args.output_dir = os.path.join(eval_output_dir, eval_task)
        eval_data_args = copy.deepcopy(data_args)
        eval_data_args.tasks = [eval_task]
        eval_data_args.eval_tasks = [eval_task]


        freezing_params(model, eval_training_args, model_args)


        trainer = T5Trainer(
          model=model,
          config=config,
          args=eval_training_args,
          train_dataset=train_dataset,
          eval_dataset=eval_dataset,
          data_collator=TaskCollator(tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores),
          compute_metrics=compute_metrics_fn[eval_task],
          data_args=eval_data_args,
          dataset_sizes=dataset_sizes,
          callbacks=[T5CheckpointCallback()]
        )
        trainer.train(
          model_path=eval_training_args.output_dir if os.path.isdir(eval_training_args.output_dir) else None
        )
        trainer.save_model()
        if trainer.is_world_process_zero():
          trainer.state.save_to_json(os.path.join(eval_training_args.output_dir, "trainer_state.json"))
          tokenizer.save_pretrained(eval_training_args.output_dir)
      else:
        trainer.model = model.to(training_args.device)
        trainer.eval_dataset = eval_dataset
        trainer.compute_metrics = compute_metrics_fn[eval_task]

      use_task_specific_params(trainer.model, eval_task)
      task_metric = trainer.evaluate()
      tasks_metric = {eval_task + "_" + k: v for k, v in task_metric.items()}
      # TODO: should it be done in word_process_zero?
      result.update(tasks_metric)
      reset_config(trainer.model, model_config)

    # logger.info(eval_datasets)
    eval_output_dir = training_args.output_dir if training_args.eval_output_dir is None else training_args.eval_output_dir
    logger.info("*** Evaluate ***")
    if trainer.is_world_process_zero():
      logger.info("***** Eval results *****")
      for key, value in result.items():
        logger.info("  %s = %s", key, value)
      save_json(result, os.path.join(eval_output_dir, "eval_results.json"))  # training_args.output_dir
      eval_results.update(result)

      # Saves the results to a gs-bucket.
      if training_args.gcs_bucket is not None:
        logger.info("***** Uploading results into gs-bucket *****")
        if training_args.eval_output_dir is not None:
          upload(training_args.eval_output_dir, training_args.gcs_bucket)
        else:
          upload(training_args.output_dir, training_args.gcs_bucket)

  if training_args.do_predict:
    logging.info("*** Test ***")

    test_output = trainer.predict(test_dataset=test_dataset)
    test_metrics = {k.replace("eval", "test"): v for k, v in test_output.metrics.items()}

    if trainer.is_world_process_zero():
      logger.info("***** Test results *****")
      for key, value in test_metrics.items():
        logger.info("  %s = %s", key, value)

      save_json(test_metrics, os.path.join(training_args.output_dir, "test_results.json"))
      eval_results.update(test_metrics)

      if training_args.predict_with_generate:
        test_preds = tokenizer.batch_decode(
          test_output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        test_preds = lmap(str.strip, test_preds)
        write_txt_file(test_preds, os.path.join(training_args.output_dir, "test_generations.txt"))

  return eval_results


def _mp_fn(index):
  # For xla_spawn (TPUs)
  main()


if __name__ == "__main__":
  main()
