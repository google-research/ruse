import collections 
import itertools
import os
import json
import pandas as pd
from tabulate import tabulate

"""
def make_name(prefix, keys, values):
  name = prefix+"-"
  for key, value in zip(keys, values):
    value = '{0:.0e}'.format(value)
    name = name+f"{key}-{value}-"
  return name[:-1]
"""
def make_name(prefix, keys, values):
  name = prefix+"-"
  for key, value in zip(keys, values):
    if isinstance(value, float):
       value = '{0:.0e}'.format(value)
    elif isinstance(value, str):
       value = value.split("/")[-1]
    #value = '{:.0e}'.format(value)
    name = name+f"{key}-{value}-"
    #name=name.replace('.', '')
  return name[:-1]

def retrieve_results(output_dir, sweep, short_keys, job_prefix, order=[]):
  print(job_prefix)
  df = pd.DataFrame()
  keys = list(sweep.keys())
  values = list(sweep.values())
  for option in list(itertools.product(*values)):
    name = make_name(job_prefix, short_keys, option)
    experiment_output_dir = os.path.join(output_dir, name)
    eval_path=os.path.join(experiment_output_dir, 'eval_results.json')
    try:
      with open(eval_path, "r") as infile:
        results = json.loads(infile.read())
      # remove losses
      results = {key: value for key, value in results.items() if "acc" in key}
      results.update({key: value for key, value in zip(keys, option)})
      df = df.append(results, ignore_index=True)
    except FileNotFoundError:
      print("File not found ", eval_path)
    

  cols = list(df.columns.values)
  for key in keys:
     cols.remove(key)
  if len(order) != 0 :
     cols = order + cols
  else:
     cols = keys + cols
  df = df[cols]
  if len(order) != 0:
     df = df.sort_values(by=order)
  #print(df.to_markdown())



  def myfunc(x):
    splits = x.split("/")
    return splits[-1].split("-")[-1]
  df['task_embedding_dir'] = df.apply(lambda x: myfunc(x.task_embedding_dir), axis=1)


  print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

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

print("mix1-finetune")
output_dir = "outputs/mixture1/finetune/"
job_prefix = "mix1-finetune"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [2e-5, 3e-3, 3e-4, 3e-5]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)

print("mix2-finetune")
output_dir = "outputs/mixture2/finetune/"
job_prefix = "mix2-finetune"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [2e-5, 3e-3, 3e-4, 3e-5]})
retrieve_results(output_dir, sweep, short_keys, job_prefix)
"""


"""
output_dir="outputs/mixture2/parametric-meta-adapter/task-emb/"
job_prefix = "m2-pmeta-task" #-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100",
                                                        "task_embeddings/n-train-1000",
                                                        "task_embeddings/n-train-2000",
                                                        "task_embeddings/n-train-all"]})
order = ["task_embedding_dir", "learning_rate"]
retrieve_results(output_dir, sweep, short_keys, job_prefix, order)

output_dir="outputs/mixture2/parametric-meta-adapter/task-emb/"
job_prefix = "m2-pmeta-task-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100",
                                                        "task_embeddings/n-train-1000",
                                                        "task_embeddings/n-train-2000",
                                                        "task_embeddings/n-train-all"]})
order = ["task_embedding_dir", "learning_rate"]
retrieve_results(output_dir, sweep, short_keys, job_prefix, order)


output_dir="outputs/mixture1/parametric-meta-adapter/task-emb/"
job_prefix = "m1-pmeta-task-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100",
                                                        "task_embeddings/n-train-1000",
                                                        "task_embeddings/n-train-2000",
                                                        "task_embeddings/n-train-all"]})
order = ["task_embedding_dir", "learning_rate"]
retrieve_results(output_dir, sweep, short_keys, job_prefix, order)


output_dir="outputs/mixture1/parametric-meta-adapter/task-emb/"
job_prefix = "m1-pmeta-task"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100",
                                                        "task_embeddings/n-train-1000",
                                                        "task_embeddings/n-train-2000",
                                                        "task_embeddings/n-train-all"]})
order = ["task_embedding_dir", "learning_rate"]
retrieve_results(output_dir, sweep, short_keys, job_prefix, order)





output_dir="outputs/mixture2/meta-adapter/task-emb"
job_prefix = "m2-meta-task"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100",
                                                        "task_embeddings/n-train-1000",
                                                        "task_embeddings/n-train-2000",
                                                        "task_embeddings/n-train-all"]})
order = ["task_embedding_dir", "learning_rate"]
retrieve_results(output_dir, sweep, short_keys, job_prefix, order)


output_dir="outputs/mixture2/meta-adapter/task-emb"
job_prefix = "m2-meta-task-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100",
                                                        "task_embeddings/n-train-1000",
                                                        "task_embeddings/n-train-2000",
                                                        "task_embeddings/n-train-all"]})
order = ["task_embedding_dir", "learning_rate"]
retrieve_results(output_dir, sweep, short_keys, job_prefix, order)



output_dir="outputs/mixture1/meta-adapter/task-emb"
job_prefix = "m1-meta-task"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100",
                                                        "task_embeddings/n-train-1000",
                                                        "task_embeddings/n-train-2000",
                                                        "task_embeddings/n-train-all"]})
order = ["task_embedding_dir", "learning_rate"]
retrieve_results(output_dir, sweep, short_keys, job_prefix, order)


output_dir="outputs/mixture1/meta-adapter/task-emb"
job_prefix = "m1-meta-task-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100",
                                                        "task_embeddings/n-train-1000",
                                                        "task_embeddings/n-train-2000",
                                                        "task_embeddings/n-train-all"]})
order = ["task_embedding_dir", "learning_rate"]
retrieve_results(output_dir, sweep, short_keys, job_prefix, order)
"""










output_dir = "outputs/mixture1/meta-adapter/task-emb/"
job_prefix = "m1-meta-task-no-relu"
#os.system("gsutil -m cp -r gs://ruse-xcloud-bucket/outputs/mixture1/meta-adapter/task-emb/"+job_prefix+"* ." )
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"]})
order = ["task_embedding_dir", "learning_rate"]
retrieve_results(output_dir, sweep, short_keys, job_prefix, order)



os.system("mkdir -p outputs/mixture2/meta-adapter/task-emb/")
job_prefix = "m2-meta-task-no-relu"
#os.system("gsutil -m cp -r gs://ruse-xcloud-bucket/outputs/mixture2/meta-adapter/task-emb/"+job_prefix+"* outputs/mixture2/meta-adapter/task-emb/" )
output_dir = "outputs/mixture2/meta-adapter/task-emb/"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"]})
retrieve_results(output_dir, sweep, short_keys, job_prefix, order)

# reorder task-embeddings.
output_dir  = "outputs/mixture1/meta-adapter/task-emb/"
job_prefix = "m1-meta-task-no-relu-reorder"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings_reordered/n-train-100"]})
retrieve_results(output_dir, sweep, short_keys, job_prefix, order)



output_dir = "outputs/mixture2/meta-adapter/task-emb/"
job_prefix = "m2-meta-task-no-relu-reorder"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings_reordered/n-train-100"]})
retrieve_results(output_dir, sweep, short_keys, job_prefix, order)
