from google.cloud import storage
from seq2seq.data import TASK_MAPPING
from logging import getLogger
import torch.nn as nn
import os

logger = getLogger(__name__)


def upload(upload_dir: str, gcs_bucket: str) -> None:
    os.system("/root/google-cloud-sdk/bin/gsutil  rm -r {}".format(os.path.join("gs://"+gcs_bucket, upload_dir)))
    os.system("/root/google-cloud-sdk/bin/gsutil -m cp -r {} {}".format(upload_dir, os.path.join("gs://"+gcs_bucket, upload_dir)))

"""
def upload(upload_dir: str, gcs_bucket: str) -> None:
  #Upload files to GCS.
  gcs_path = upload_dir
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(gcs_bucket)
  for dirpath, _, filenames in os.walk(upload_dir):
    for name in filenames:
      filename = os.path.join(dirpath, name)
      blob = storage.Blob(os.path.join(gcs_path, name), bucket)
      with open(filename, 'rb') as f:
        blob.upload_from_file(f, num_retries=10, timeout=10*60)
"""

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
    # TODO(rabeeh): unfreezed patterns need to be a list.
    """Freezes all the parameters of the model expect for the specified not_freezed_pattern."""
    for name, p in model.named_parameters():
        if not_freezed_pattern in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
        #p.requires_grad = True if not_freezed_pattern in name else False




