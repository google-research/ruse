import os
import json
import collections
import itertools
import copy


job_string_base = '''
#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l q_short_gpu  -l h=vgn[ehgf]* -P nlu 

source activate internship
'''

def run_jobs_local(config_path, name, outpath="/idiap/temp/rkarimi/task-emb"):
    if not os.path.exists(outpath):
       os.makedirs(outpath)
    curr_job=job_string_base+'''python compute_task_emb.py  {0}'''.format(config_path)
    job_name = "template.job"
    with open(job_name, "w") as f:
       f.write(curr_job)
    os.system("qsub -V -N {0} -e {1}.err -o {1}.out template.job".format(name, os.path.join(outpath, name)))


def run_jobs(config_path, job_name):
  command = "/google/bin/releases/cloud-alphabetcloud-xcloud/xcloud_cli/xcloud_cli.par google/launch_xla_clean1.py  -- --config_path {0} --job_name {1} --num_gpus 1".format(
    config_path, job_name)
  os.system(command)

def make_name(prefix, keys, values):
  name = prefix+"-"
  for key, value in zip(keys, values):
    value = '{0:.0e}'.format(value)
    #value = '{:.0e}'.format(value)
    name = name+f"{key}-{value}-"
    #name=name.replace('.', '')
  return name[:-1]

def do_sweep(basic_config_path, sweep, short_keys, job_prefix, local=False):
  with open(basic_config_path, "r") as infile:
    parent_config = json.loads(infile.read())
  values = list(sweep.values())
  keys = list(sweep.keys())
  for option in list(itertools.product(*values)):
    config = copy.deepcopy(parent_config)
    updates = {key: value for key, value in zip(keys, option)}
    config.update(updates)
    name = make_name(job_prefix, short_keys, option)
    print("### name ", name)
    output_dir = os.path.join(parent_config['output_dir'], name)
    config.update({'output_dir': output_dir})
    config_path = "temp.json"
    with open(config_path, 'w') as f:
      json.dump(config, f)
    if local:
       run_jobs_local(config_path, name)
    else:
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

# compute task embeddings for different test samples.
basic_config_path = "configs/task-embedding.json"
short_keys = ["n_train"]
job_prefix="task-emb"
sweep = collections.OrderedDict({'n_train': [100, 500, 1000, 2000, 3000]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, local=True)

