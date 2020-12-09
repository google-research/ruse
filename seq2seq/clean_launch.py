import os
import json
import collections
import itertools
import copy

def run_jobs(config_path, job_name):
  command = "/google/bin/releases/cloud-alphabetcloud-xcloud/xcloud_cli/xcloud_cli.par google/launch_xla_clean.py  -- --config_path {0} --job_name {1} --num_gpus 1".format(
    config_path, job_name)
  os.system(command)

def make_name(prefix, keys, values):
  name = prefix+"-"
  for key, value in zip(keys, values):
    name = name+f"{key}-{value}-"
  return name[:-1]

def do_sweep(basic_config_path, sweep, short_keys, job_prefix):
  with open(basic_config_path, "r") as infile:
    parent_config = json.loads(infile.read())
  values = list(sweep.values())
  keys = list(sweep.keys())
  for option in list(itertools.product(*values)):
    config = copy.deepcopy(parent_config)
    updates = {key: value for key, value in zip(keys, option)}
    config.update(updates)
    name = make_name(job_prefix, short_keys, option)
    output_dir = os.path.join(parent_config['output_dir'], name)
    config.update({'output_dir': output_dir})
    config_path = "temp.json"
    with open(config_path, 'w') as f:
      json.dump(config, f)
    run_jobs(config_path, name)


# Mixture 1.
basic_config_path="configs/experiments/mixture1/meta-rand.json"
job_prefix = "mix1-meta-rand"
short_keys = ["lr"] #, "rate"]
sweep = collections.OrderedDict({'learning_rate': [1e-2]}) #, 3e-1, 3e-2, 3e-3, 3e-4]}) #, 'rate': [5, 10, 15]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)

"""
basic_config_path="configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "mix1-meta-task-emb"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path="configs/experiments/mixture1/paramteric-meta-rand.json"
job_prefix = "mix1-param-meta-rand"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path="configs/experiments/mixture1/paramteric-meta-task-emb.json"
job_prefix = "mix1-param-meta-task-emb"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


# Mixture 2.
basic_config_path="configs/experiments/mixture2/meta-rand.json"
job_prefix = "mix2-meta-rand"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path="configs/experiments/mixture2/meta-task-emb.json"
job_prefix = "mix2-meta-task-emb"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path="configs/experiments/mixture2/paramteric-meta-rand.json"
job_prefix = "mix2-param-meta-rand"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path="configs/experiments/mixture2/paramteric-meta-task-emb.json"
job_prefix = "mix2-param-meta-task-emb"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""
