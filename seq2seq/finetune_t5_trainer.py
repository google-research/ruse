import logging
import os
import sys

import datasets
from seq2seq.trainers import T5Trainer
from seq2seq.training_args import Seq2SeqTrainingArguments, ModelArguments, DataTrainingArguments
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser, MBartTokenizer, set_seed
from transformers.trainer_utils import EvaluationStrategy
from seq2seq.utils import (
    assert_all_frozen,
    check_output_dir,
    freeze_embeds,
    freeze_params,
    partly_freeze_params,
    lmap,
    save_json,
    write_txt_file,
)
from seq2seq.models import T5Config
from seq2seq.tasks import AutoTask, TaskCollator
from seq2seq.metrics import build_compute_metrics_fn
from seq2seq.samplers import compute_data_groups
from seq2seq.models import T5ForConditionalGeneration

import torch.nn as nn

from seq2seq.adapters import Adapter
from transformers.modeling_t5  import T5LayerNorm

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:

        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else\
          model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if model_args.not_load_t5_checkpoint:
        model = T5ForConditionalGeneration(config=config, tasks=data_args.task)
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            tasks=data_args.task
        )

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    if training_args.train_adapters:
        # TODO: check this is working.
        partly_freeze_params(model, 'adapter')
        # Sets the last layer of decoder to be trained.
        for param in model.lm_head.parameters():
          param.require_grad = True

        # TODO(rabeeh): to be checked if needed.
        #for name, sub_module in model.named_modules():
        #    if isinstance(sub_module, (Adapter, T5LayerNorm, nn.LayerNorm)):
        #        print("#### sub_module ", sub_module)
        #        for param_name, param in sub_module.named_parameters():
        #            param.requires_grad = True
    else:
        if model_args.freeze_embeds:
            freeze_embeds(model)
        if model_args.freeze_encoder:
            freeze_params(model.get_encoder())
            assert_all_frozen(model.get_encoder())

    dataset_class = AutoTask
    if training_args.do_train:
        train_dataset = {task: dataset_class.get(task).get_dataset(
            split="train", n_obs=data_args.n_train) for task in data_args.task}
        # TODO: can be reordered later on.
        #data_groups = compute_data_groups(list(train_dataset.values()))
        #train_dataset = datasets.concatenate_datasets(list(train_dataset.values()))
    else:
        # TODO; for now lets concat the datasets.
        train_dataset = None


    # TODO: you should not do this, introduces bug.
    #train_dataset.set_format(type="torch", columns=['src_texts', 'tgt_texts'])
    training_args.remove_unused_columns = False

    # TODO: split varies.
    eval_datasets = ({task: dataset_class.get(task).get_dataset(
        split="validation", n_obs=data_args.n_val) for task in data_args.task}
        if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )
    test_dataset = (
        {task: dataset_class.get(task).get_dataset(
            split="test", n_obs=data_args.n_test) for task in data_args.task}
        if training_args.do_predict
        else None
    )

    # TODO: this needs to get fixed, for now we do not need it.
    # Initialize our Trainer
    compute_metrics_fn = (
        build_compute_metrics_fn(data_args.task, tokenizer) if training_args.predict_with_generate else None
    )

    # TODO: how does it get between different max_lengths?
    trainer = T5Trainer(
        model=model,
        config=config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # Since prototype does not match we feed this in later. #eval_dataset,
        data_collator=TaskCollator(tokenizer, data_args, training_args.tpu_num_cores),
        compute_metrics=compute_metrics_fn,
        data_args=data_args,
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
        training_args.output_dir, # "t5-base" for the baseline.
        cache_dir=model_args.cache_dir,
        )
        # TODO: using task-specific params, should be set globally during eval.
        model = T5ForConditionalGeneration.from_pretrained(
            training_args.output_dir, # "t5-base" for the baseline.
            from_tf=".ckpt" in training_args.output_dir,
            config=config,
            cache_dir=model_args.cache_dir,
            tasks=data_args.task
        )
        trainer.model = model.to(training_args.device)

        logger.info(eval_datasets)
        logger.info("*** Evaluate ***")

        result = trainer.evaluate(eval_datasets, compute_metrics_fn)
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
