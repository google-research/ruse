"""
A simple launcher script for TPU training

Inspired by https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py

::
    >>> python xla_spawn.py --num_cores=NUM_CORES_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

"""


import importlib
import sys
from argparse import REMAINDER, ArgumentParser
from pathlib import Path
import json
from google.cloud import storage
import os

from transformers.file_utils import is_torch_tpu_available
if is_torch_tpu_available():
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met



def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description=(
            "PyTorch TPU distributed training launch "
            "helper utility that will spawn up "
            "multiple distributed processes"
        )
    )

    # Optional arguments for the launch helper
    parser.add_argument("--num_cores", type=int, default=8, help="Number of TPU cores to use (1 or 8).")

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help=(
            "The full path to the single TPU training "
            "program/script to be launched in parallel, "
            "followed by all the arguments for the "
            "training script"
        ),
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)

    return parser.parse_args()


def upload(upload_dir: str, gcs_bucket: str, gcs_path: str = None) -> None:
  """Upload files to GCS.
  """
  gcs_path = upload_dir
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(gcs_bucket)
  for dirpath, _, filenames in os.walk(upload_dir):
    for name in filenames:
      filename = os.path.join(dirpath, name)
      blob = storage.Blob(os.path.join(gcs_path, name), bucket)
      with open(filename, 'rb') as f:
        blob.upload_from_file(f)


def main():
    args = parse_args()
    # Import training_script as a module.
    script_fpath = Path(args.training_script)
    sys.path.append(str(script_fpath.parent.resolve()))
    mod_name = script_fpath.stem
    mod = importlib.import_module(mod_name)

    # Patch sys.argv
    sys.argv = [args.training_script] + args.training_script_args #+ ["--tpu_num_cores", str(args.num_cores)]


    xmp.spawn(mod._mp_fn, args=(), nprocs=args.num_cores)
    #xm.rendezvous('done')


    #training_dict = json.loads( args.training_script_args)
    #gcs_bucket="gs://ruse-xcloud-bucket/"
    #os.system("gsutil cp -r {} {}".format(training_dict['output_dir'],
    #  os.path.join(gcs_bucket, training_dict['output_dir'])))
    gcs_bucket="ruse-xcloud-bucket"
    with open(args.training_script_args[0], "r") as infile:
    	data = json.loads(infile.read())
    upload_dir= data['output_dir']
    upload(upload_dir, gcs_bucket)


if __name__ == "__main__":
    main()
