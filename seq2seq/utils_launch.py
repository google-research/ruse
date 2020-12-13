import copy
import itertools
import os
import json
import pandas as pd
from tabulate import tabulate


def run_jobs(config_path, job_name):
  command = "/google/bin/releases/cloud-alphabetcloud-xcloud/xcloud_cli/xcloud_cli.par google/launch_xla_clean1.py  -- --config_path {0} --job_name {1} --num_gpus 1".format(
    config_path, job_name)
  os.system(command)

def flatten(output):
  flatten = []
  for a in output:
    if type(a) == tuple:
      flatten.extend(list(a))
    else:
      flatten.append(a)
  return flatten

def make_name(prefix, keys, values):
  name = prefix+"-"
  for key, value in zip(keys, values):
    if isinstance(value, float):
       value = '{0:.0e}'.format(value)
    elif isinstance(value, str):
       value = value.split("/")[-1]
    name = name+f"{key}-{value}-"
    name = name.lower()
  return name[:-1]


def do_sweep(parent_config_path, sweep, short_keys, job_prefix, output_dir_name="output_dir"):
  with open(parent_config_path, "r") as infile:
    parent_config = json.loads(infile.read())
  values = list(sweep.values())
  keys = flatten(list(sweep.keys()))
  options = [flatten(option) for option in list(itertools.product(*values))]
  for option in options:
    config = copy.deepcopy(parent_config)
    config.update({key: value for key, value in zip(keys, option)})
    name = make_name(job_prefix, short_keys, option)
    print("### name ", name)
    if output_dir_name in parent_config:
      parent_output_dir = parent_config[output_dir_name]
    else:
      parent_output_dir = sweep[output_dir_name][0]
    output_dir = os.path.join(parent_output_dir, name)
    config.update({output_dir_name: output_dir})
    config_path = "temp.json"
    with open(config_path, 'w') as f:
      json.dump(config, f)
    run_jobs(config_path, name)

def myfunc(x):
    splits = x.split("/")
    return splits[-1].split("-")[-1]

acc_cols = ['cola_eval_acc',   'snli_eval_acc', 'yelp_polarity_eval_acc']
def retrieve_results(output_dir, sweep, short_keys, job_prefix, params=[]):
  print(job_prefix)
  df = pd.DataFrame()
  keys = flatten(list(sweep.keys()))
  values = list(sweep.values())
  options = [flatten(option) for option in list(itertools.product(*values))]
  for option in options:
    name = make_name(job_prefix, short_keys, option)
    experiment_output_dir = os.path.join(output_dir, name)
    eval_path = os.path.join(experiment_output_dir, 'eval_results.json')
    try:
      with open(eval_path, "r") as infile:
        results = json.loads(infile.read())
      results.update({key: value for key, value in zip(keys, option)})
      df = df.append(results, ignore_index=True)
    except FileNotFoundError:
      print("File not found ", eval_path)
  df = df[params+acc_cols]
  if len(params) != 0:
    df = df.sort_values(by=params)
  #df['task_embedding_dir'] = df.apply(lambda x: myfunc(x.task_embedding_dir), axis=1)
  print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
