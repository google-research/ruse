import collections 
import itertools
import os
import json
import pandas as pd
from tabulate import tabulate

def make_name(prefix, keys, values):
  name = prefix+"-"
  for key, value in zip(keys, values):
    value = '{0:.0e}'.format(value)
    name = name+f"{key}-{value}-"
  return name[:-1]

def retrieve_results(output_dir, sweep, short_keys, job_prefix):
  df = pd.DataFrame()
  keys = list(sweep.keys())
  values = list(sweep.values())
  for option in list(itertools.product(*values)):
    name = make_name(job_prefix, short_keys, option)
    experiment_output_dir = os.path.join(output_dir, name)
    with open(os.path.join(experiment_output_dir, 'eval_results.json'), "r") as infile:
      results = json.loads(infile.read())

    # remove losses
    results = {key: value for key, value in results.items() if "acc" in key}
    results.update({key: value for key, value in zip(keys, option)})
    df = df.append(results, ignore_index=True)

  cols = list(df.columns.values)
  cols.remove('learning_rate')
  cols = ['learning_rate'] + cols
  df = df[cols]
  #print(df.to_markdown())
  print(tabulate(df, headers='keys', tablefmt='rst', showindex=False))
  #print(output.head())


"""
output_dir= "outputs/mixture1/meta-adapter/rand/"
job_prefix = "mix1-meta-rand"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]}) 
retrieve_results(output_dir, sweep, short_keys, job_prefix)


output_dir= "outputs/mixture2/meta-adapter/rand/"
job_prefix = "mix2-meta-rand"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]}) 
retrieve_results(output_dir, sweep, short_keys, job_prefix)

output_dir= "outputs/mixture1/meta-adapter/task-emb/"
job_prefix = "mix1-meta-task-emb"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]}) 
retrieve_results(output_dir, sweep, short_keys, job_prefix)

output_dir= "outputs/mixture2/meta-adapter/task-emb/"
job_prefix = "mix2-meta-task-emb"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]}) 
retrieve_results(output_dir, sweep, short_keys, job_prefix)

output_dir= "outputs/mixture1/parametric-meta-adapter/rand/"
job_prefix = "mix1-param-meta-rand"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)


output_dir= "outputs/mixture2/parametric-meta-adapter/rand/"
job_prefix = "mix2-param-meta-rand"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)


output_dir= "outputs/mixture1/parametric-meta-adapter/task-emb/"
job_prefix = "mix1-param-meta-task-emb"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)


output_dir= "outputs/mixture2/parametric-meta-adapter/task-emb/"
job_prefix = "mix2-param-meta-task-emb"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)


output_dir= "outputs/mixture1/parametric-meta-adapter-with-lm-head/rand/"
job_prefix = "mix1-param-meta-rand-head"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)


output_dir= "outputs/mixture2/parametric-meta-adapter-with-lm-head/rand/"
job_prefix = "mix2-param-meta-rand-head"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)


output_dir= "outputs/mixture1/only-lm-head/"
job_prefix = "mix1-only-head"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)


output_dir= "outputs/mixture2/only-lm-head/"
job_prefix = "mix2-only-head"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)


output_dir="outputs/mixture1/t5-scratch/"
job_prefix = "mix1-t5-scratch"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)

output_dir="outputs/mixture2/t5-scratch/"
job_prefix = "mix2-t5-scratch"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)
"""


print("mix1-meta-task-emb-new")
output_dir = "outputs/mixture1/meta-adapter/task-emb/"
job_prefix = "mix1-meta-task-emb-new"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)

print("mix2-meta-task-emb-new")
output_dir = "outputs/mixture2/meta-adapter/task-emb/"
job_prefix = "mix2-meta-task-emb-new"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)


print("mix1-meta-task-emb-no-ln-r")
output_dir = "outputs/mixture1/meta-adapter-no-layer-norm/task-emb/"
job_prefix = "mix1-meta-task-emb-no-ln-r"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)

print("mix2-meta-task-emb-no-ln-r")
output_dir = "outputs/mixture2/meta-adapter-no-layer-norm/task-emb/"
job_prefix = "mix2-meta-task-emb-no-ln-r"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)


print("mix1-meta-task-false-true-r")
output_dir = "outputs/mixture1/meta-adapter-no-layer-norm-inside-pre-false-post-true/task-emb/"
job_prefix = "mix1-meta-task-false-true-r"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)

print("mix2-meta-task-false-true-r")
output_dir = "outputs/mixture2/meta-adapter-no-layer-norm-inside-pre-false-post-true/task-emb/"
job_prefix = "mix2-meta-task-false-true-r"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)

print("mix1-meta-task-true-false")
output_dir = "outputs/mixture1/meta-adapter-no-layer-norm-inside-pre-true-post-false/task-emb/"
job_prefix = "mix1-meta-task-true-false"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)

print("mix2-meta-task-true-false")
output_dir = "outputs/mixture2/meta-adapter-no-layer-norm-inside-pre-true-post-false/task-emb/"
job_prefix = "mix2-meta-task-true-false"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)

