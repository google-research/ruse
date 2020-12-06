import logging
import os
import sys

import datasets
from transformers.file_utils import is_torch_tpu_available
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import EvaluationStrategy
from seq2seq.utils import (
    assert_all_frozen,
    check_output_dir,
    freeze_embeds,
    freeze_params,
    lmap,
    save_json,
    write_txt_file,
)

from seq2seq.trainers import T5Trainer
from seq2seq.training_args import Seq2SeqTrainingArguments, ModelArguments, DataTrainingArguments, AdapterTrainingArguments
from seq2seq.models import T5Config
from seq2seq.tasks import AutoTask, TaskCollator
from seq2seq.metrics import build_compute_metrics_fn
from seq2seq.models import T5ForConditionalGeneration
from seq2seq.adapters import AdapterController, MetaAdapterController, MetaParamterizedAdapterController, AutoAdapterConfig


logger = logging.getLogger(__name__)

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm


def shard_data(datasets, num_replicas, rank):
    """Returns the sharded data belonging to the given rank."""
    for i, dataset in enumerate(datasets):
        sharded_dataset = dataset.shard(num_replicas, rank)
        datasets[i] = sharded_dataset
    return datasets

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
                          # ,
                          #"meta_adapters", "task_embedding_dir",
                          #"meta_parameterized_adapters")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))

    # Gets the adapter config and updates the specified parameters.
    adapter_config = AutoAdapterConfig.get(adapter_args.adapter_config_name)
    adapter_config.input_dim = config.d_model
    adapter_config.tasks = data_args.tasks
    extra_adapter_params = ("task_embedding_dir")
    for p in extra_adapter_params:
        if getattr(adapter_args, p, None):
            assert hasattr(adapter_config, p), f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(adapter_config, p, getattr(adapter_args, p))
    #setattr(config, "adapter_config", adapter_config)


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else\
          model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    print("#### inside main ", adapter_config)
    if model_args.not_load_t5_checkpoint:
        model = T5ForConditionalGeneration(config=config, adapter_config=adapter_config) #, tasks=data_args.tasks)
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            adapter_config=adapter_config
            #tasks=data_args.tasks
        )

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    if training_args.train_adapters:
        # Sets the last layer of decoder to be trained.
        freeze_params(model)
        for param in model.lm_head.parameters():
          param.require_grad = True
        #if training_args.meta_adapters:
        # Sets the gradient for all meta-adapters to True.
        for name, sub_module in model.named_modules():
           if isinstance(sub_module, (MetaAdapterController, MetaParamterizedAdapterController)):
              for param_name, param in sub_module.named_parameters():
                 param.requires_grad = True
            
    else:
        if model_args.freeze_embeds:
            freeze_embeds(model)
        if model_args.freeze_encoder:
            freeze_params(model.get_encoder())
            assert_all_frozen(model.get_encoder())

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
    #train_dataset.set_format(type="torch", columns=['src_texts', 'tgt_texts'])
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
        eval_dataset=eval_datasets, #None,  # Since prototype does not match we feed this in later. #eval_dataset,
        data_collator=TaskCollator(tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores), #, tasks=data_args.tasks),
        compute_metrics=compute_metrics_fn,
        data_args=data_args,
        dataset_sizes=dataset_sizes if training_args.do_train else None,
        task_to_compute_metrics = compute_metrics_fn
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
        config = T5Config.from_pretrained(
        training_args.output_dir,# "t5-base" for the baseline.
        cache_dir=model_args.cache_dir)
        # TODO: using task-specific params, should be set globally during eval.
        model = T5ForConditionalGeneration.from_pretrained(
            training_args.output_dir, # "t5-base" for the baseline.
            from_tf=".ckpt" in training_args.output_dir,
            config=config,
            cache_dir=model_args.cache_dir,
            adapter_config=adapter_config
        )
        trainer.model = model.to(training_args.device)

        if training_args.train_adapters:
            # If task to adapter is given set it in all adapter controller layers.
            if data_args.adapters is not None:
                task_to_adapter = {eval_task: adapter for eval_task, adapter in
                                   zip(data_args.eval_tasks, data_args.adapters)}
                for name, sub_module in model.named_modules():
                   # TODO: they need to be object of the same class.
                   if isinstance(sub_module, AdapterController) or \
                       isinstance(sub_module, MetaAdapterController):
                       sub_module.set_task_to_adapter_map(task_to_adapter)

        logger.info(eval_datasets)
        logger.info("*** Evaluate ***")

        result = trainer.evaluate() #eval_datasets, compute_metrics_fn)
        if trainer.is_world_process_zero():
            logger.info("***** Eval results *****")
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
            save_json(result, os.path.join(training_args.output_dir, "eval_results.json"))
            eval_results.update(result)

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
