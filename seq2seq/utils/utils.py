import os
import torch.nn as nn
from google.cloud import storage
from logging import getLogger
from third_party.utils import (
    assert_all_frozen,
    freeze_embeds,
    freeze_params)
from transformers import TrainerCallback
from transformers.modeling_t5 import T5LayerNorm

from seq2seq.adapters import AdapterController, MetaAdapterController
from seq2seq.data import TASK_MAPPING

logger = getLogger(__name__)


def upload(upload_dir: str, gcs_bucket: str) -> None:
    """Uploads the local upload_dir to the gs bucket."""
    os.system("/root/google-cloud-sdk/bin/gsutil  rm -r {}".format(
        os.path.join("gs://" + gcs_bucket, upload_dir)))
    os.system("/root/google-cloud-sdk/bin/gsutil -m cp -r {} {}".format(
        upload_dir,
        os.path.join("gs://" + gcs_bucket, upload_dir)))


def upload_with_storage(upload_dir: str, gcs_bucket: str) -> None:
    """Uploads the upload_dir to the gs_bucket. Note that this method could
    results in timeout issues and is better not to use storage library and
    only rely on gsutil for copying files."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(gcs_bucket)
    for dirpath, _, filenames in os.walk(upload_dir):
        for name in filenames:
            filename = os.path.join(dirpath, name)
            blob = storage.Blob(os.path.join(upload_dir, name), bucket)
            with open(filename, 'rb') as f:
                blob.upload_from_file(f, num_retries=10, timeout=10 * 60)


def use_task_specific_params(model, task):
    """Update config with task specific params during evaluation."""
    task_dataset = TASK_MAPPING[task]
    task_specific_config = task_dataset.task_specific_config
    if task_specific_config is not None:
        logger.info(f"using task specific params for {task}: {task_specific_config}")
        model.config.update(task_specific_config)


def reset_config(model, config):
    """Resets the config file to the one provided."""
    model.config = config
    logger.info(f"config is reset to the initial values.")


def use_task_specific_params(model, task):
    """Update config with task specific params during evaluation."""
    task_dataset = TASK_MAPPING[task]
    task_specific_config = task_dataset.task_specific_config
    if task_specific_config is not None:
        logger.info(f"using task specific params for {task}: {task_specific_config}")
        model.config.update(task_specific_config)


def reset_config(model, config):
    """Resets the config file to the one provided."""
    model.config = config
    logger.info(f"config is reset to the initial values.")


def partly_freeze_params(model: nn.Module, not_freezed_pattern):
    """Freezes all the parameters of the model expect for the specified not_freezed_pattern."""
    for name, p in model.named_parameters():
        if not_freezed_pattern in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


def freezing_params(model, training_args, model_args):
    if training_args.train_adapters:
        freeze_params(model)
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (MetaAdapterController, AdapterController)):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True
        for param in model.task_embedding_controller.parameters():
            param.requires_grad = True

    if model_args.freeze_model:
        freeze_params(model)

    if model_args.freeze_model_but_lm_head:
        freeze_params(model)
        for param in model.lm_head.parameters():
            param.requires_grad = True

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


class T5CheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        """Event called after a checkpoint save."""
        if state.is_world_process_zero and args.gcs_bucket is not None:
            upload(args.output_dir, args.gcs_bucket)
