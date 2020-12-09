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
  print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
  #print(output.head())


output_dir= "outputs/mixture1/meta-adapter/rand/"
job_prefix = "mix1-meta-rand"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]}) 
retrieve_results(output_dir, sweep, short_keys, job_prefix)


