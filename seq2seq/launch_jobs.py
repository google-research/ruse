import os
import json
import collections
import itertools
import copy


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


"""
basic_config_path="configs/experiments/mixture1/test.json"
job_prefix = "test"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

"""
# Mixture 1.
basic_config_path="configs/experiments/mixture1/meta-rand.json"
job_prefix = "mix1-meta-rand"
short_keys = ["lr"] #, "rate"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]}) #, 'rate': [5, 10, 15]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)

basic_config_path="configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "mix1-meta-task-emb" # -new
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
job_prefix = "mix2-meta-task-emb" # -new
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

"""
basic_config_path="configs/experiments/mixture1/paramteric-meta-rand-with-lm-head.json"
job_prefix = "mix1-param-meta-rand-head"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path="configs/experiments/mixture2/paramteric-meta-rand-with-lm-head.json"
job_prefix = "mix2-param-meta-rand-head"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

"""
basic_config_path="configs/experiments/mixture1/only-lm-head.json"
job_prefix = "mix1-only-head"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path="configs/experiments/mixture2/only-lm-head.json"
job_prefix = "mix2-only-head"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""


"""
basic_config_path="configs/experiments/mixture1/t5-scratch.json"
job_prefix = "mix1-t5-scratch"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path = "configs/experiments/mixture2/t5-scratch.json"
job_prefix = "mix2-t5-scratch"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

"""
basic_config_path="configs/experiments/mixture1/meta-task-emb-no-layer-norm.json"
job_prefix = "mix1-meta-task-emb-no-ln-r"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path="configs/experiments/mixture2/meta-task-emb-no-layer-norm.json"
job_prefix = "mix2-meta-task-emb-no-ln-r"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)

basic_config_path="configs/experiments/mixture1/meta-task-emb-no-layer-norm-layernorm-inside-controller-pre-false-post-true.json"
job_prefix = "mix1-meta-task-false-true-r"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)

basic_config_path="configs/experiments/mixture1/meta-task-emb-no-layer-norm-layernorm-inside-controller-pre-true-post-false.json"
job_prefix = "mix1-meta-task-true-false-r"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)

basic_config_path="configs/experiments/mixture1/meta-task-emb-no-layer-norm-layernorm-inside-controller-pre-true-post-true.json"
job_prefix = "mix1-meta-task-true-true-r"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)

basic_config_path="configs/experiments/mixture2/meta-task-emb-no-layer-norm-layernorm-inside-controller-pre-false-post-true.json"
job_prefix = "mix2-meta-task-false-true-r"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)

basic_config_path="configs/experiments/mixture2/meta-task-emb-no-layer-norm-layernorm-inside-controller-pre-true-post-false.json"
job_prefix = "mix2-meta-task-true-false-r"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)

basic_config_path="configs/experiments/mixture2/meta-task-emb-no-layer-norm-layernorm-inside-controller-pre-true-post-true.json"
job_prefix = "mix2-meta-task-true-true-r"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""


"""
# finetune.
basic_config_path="configs/experiments/mixture1/finetune.json"
job_prefix = "mix1-finetune"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [2e-5, 3e-3, 3e-4, 3e-5]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)

basic_config_path="configs/experiments/mixture2/finetune.json"
job_prefix = "mix2-finetune"
short_keys = ["lr"]
sweep = collections.OrderedDict({'learning_rate': [2e-5, 3e-3, 3e-4, 3e-5]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

"""
basic_config_path = "configs/experiments/mixture2/paramteric-meta-task-emb.json"
job_prefix = "m2-pmeta-task-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100",
                                                        "task_embeddings/n-train-1000",
                                                        "task_embeddings/n-train-2000",
                                                        "task_embeddings/n-train-all"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)

basic_config_path = "configs/experiments/mixture2/meta-task-emb.json"
job_prefix = "m2-meta-task-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100",
                                                        "task_embeddings/n-train-1000",
                                                        "task_embeddings/n-train-2000",
                                                        "task_embeddings/n-train-all"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)





basic_config_path = "configs/experiments/mixture1/paramteric-meta-task-emb.json"
job_prefix = "m1-pmeta-task-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100",
                                                        "task_embeddings/n-train-1000",
                                                        "task_embeddings/n-train-2000",
                                                        "task_embeddings/n-train-all"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-meta-task-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100",
                                                        "task_embeddings/n-train-1000",
                                                        "task_embeddings/n-train-2000",
                                                        "task_embeddings/n-train-all"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-meta-task-no-relu"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

"""
basic_config_path = "configs/experiments/mixture2/meta-task-emb.json"
job_prefix = "m2-meta-task-no-relu"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

"""
# reorder task-embeddings.
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-meta-task-norel-re"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings_reordered/n-train-100"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path = "configs/experiments/mixture2/meta-task-emb.json"
job_prefix = "m2-meta-task-norel-re"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings_reordered/n-train-100"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "test1"
short_keys = ["n", "lr", "e"]
sweep = collections.OrderedDict({'n_finetune': [1], #, 500, 1000, 2000, 4000],
                                 'learning_rate': [1e-2], #, 3e-1, 3e-2, 3e-3, 3e-4],
                                 "num_train_epochs": [2], #, 100, 200],
                                 "do_finetune": [True],
                                 "do_train":[True],
                                 "n_train": [10],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["outputs/test"],
                                 "eval_output_dir": ["outputs/test/eval/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""

"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-ftune-adapter"
short_keys = ["n", "lr", "e"]
sweep = collections.OrderedDict({'n_finetune': [100, 500, 1000, 2000, 4000],
                                 'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 "num_train_epochs": [2000, 10000, 20000],
                                 "do_finetune": [True],
                                 "do_train":[False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-meta-task-no-relu-lr-3e-02-emb-n-train-100"],
                                 "eval_output_dir": ["outputs/finetune-adapter/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")

basic_config_path = "configs/experiments/mixture2/meta-task-emb.json"
job_prefix = "m2-ftune-adapter"
short_keys = ["n", "lr", "e"]
sweep = collections.OrderedDict({'n_finetune': [100, 500, 1000, 2000, 4000],
                                 'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 "num_train_epochs": [20, 100, 200],
                                 "do_finetune": [True],
                                 "eval_tasks": [["qnli", "scitail", "boolq"]],
                                 "do_train":[False],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m2-meta-task-no-relu-lr-3e-02-emb-n-train-100"],
                                 "eval_output_dir": ["outputs/finetune-adapter/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""



"""
# finetuning both models with different number of samples for steps=140000.
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-adp-v"
short_keys = ["lr", "n", "e", "h"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [8960, 1792, 896, 448, 224]),
                                 "unfreeze_lm_head": [True, False],
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-meta-task-no-relu-lr-3e-02-emb-n-train-100"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-adapter/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")



basic_config_path = "configs/experiments/mixture1/finetune.json"
job_prefix = "m1-t5-v"
short_keys = ["lr", "n", "e"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [8960, 1792, 896, 448, 224]),
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "output_dir": ["mix1-finetune-lr-3e-04"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-t5/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""



# let finetune the lm-head
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-meta-task-no-relu-lm"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 "unfreeze_lm_head": [True],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path = "configs/experiments/mixture2/meta-task-emb.json"
job_prefix = "m2-meta-task-no-relu-lm"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 "unfreeze_lm_head": [True],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
