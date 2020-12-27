import copy
import itertools
import json
import pandas as pd
from tabulate import tabulate
import subprocess
import os

def get_run_command(config_path, job_name, num_gpus):
  with open('google/launch_command', 'r') as f:
    launch_command = f.read()
  launch_command = launch_command.rstrip()
  return "{0} google/launch_xla_clean1.py -- --config_path {1} --job_name {2} --num_gpus {3} --norun_copybara --noxc_monitor_on_launch".format(launch_command, config_path, job_name, num_gpus)
  

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


def do_sweep(parent_config_path, sweep,  job_prefix, short_keys=None, output_dir_name="output_dir",
             failed_jobs=[], num_gpus=1, ordered_name=False):
  temp_configs_dir = "temp_configs"
  if not os.path.exists(temp_configs_dir):
     os.makedirs(temp_configs_dir)
  with open(parent_config_path, "r") as infile:
    parent_config = json.loads(infile.read())
  values = list(sweep.values())
  keys = flatten(list(sweep.keys()))
  options = [flatten(option) for option in list(itertools.product(*values))]

  i = 0
  commands = []
  for option in options:
    config = copy.deepcopy(parent_config)
    config.update({key: value for key, value in zip(keys, option)})
    if ordered_name:
        name = job_prefix+"-"+str(i)
        i = i + 1
    else:
        name = make_name(job_prefix, short_keys, option)

    print("### name ", name)
    if output_dir_name in sweep: #parent_config:
      parent_output_dir = sweep[output_dir_name][0]
    else:
      parent_output_dir = parent_config[output_dir_name]
    output_dir = os.path.join(parent_output_dir, name)
    config.update({output_dir_name: output_dir})
    config_path = "{0}/{1}.json".format(temp_configs_dir, name)
    with open(config_path, 'w') as f:
      json.dump(config, f)
    if (len(failed_jobs) != 0 and name in failed_jobs) or len(failed_jobs) == 0:
      commands.append(get_run_command(config_path, name, num_gpus))
  print(len(commands))
  run_in_parallel(commands)

  

def myfunc(x):
    splits = x.split("/")
    return splits[-1].split("-")[-1]


def run_in_parallel(commands):
  processes = set()
  max_processes = 4
  for name in commands:
    processes.add(subprocess.Popen(name, shell=True, stdout=subprocess.PIPE))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update(
            [p for p in processes if p.poll() is not None])
  #Check if all the child processes were closed
  for p in processes:
    if p.poll() is None:
        p.wait()


def copy_command(experiment_output_dir, bucket, copy_commands, file_name="eval_results.json"):
        eval_path = os.path.join(experiment_output_dir, file_name)
        if not os.path.exists(experiment_output_dir):
            os.makedirs(experiment_output_dir)
        if not os.path.exists(eval_path):
            copy_commands.append(f"gsutil cp {bucket}/{eval_path} {experiment_output_dir}/{file_name}")
            print(copy_commands)

def download_all_evals(sweep, job_prefix, short_keys, output_dir, ordered_name=False):
  values = list(sweep.values())
  options = [flatten(option) for option in list(itertools.product(*values))]
  bucket = "gs://ruse-xcloud-bucket"
  copy_commands = []
  i = 0 
  for option in options:
    if ordered_name:
        name = job_prefix+"-"+str(i)
        i = i + 1
    else:
        name = make_name(job_prefix, short_keys, option)
    experiment_output_dir = os.path.join(output_dir, name)
    if ordered_name:
        copy_command(experiment_output_dir, bucket, copy_commands, file_name="test_results.json")
        copy_command(experiment_output_dir, bucket, copy_commands, file_name="val_results.json")
        copy_command(experiment_output_dir, bucket, copy_commands, file_name="arguments_results.json")
        #download("test_results.json", "val_results.json", "arguments_results.json")
    else:
        copy_command(experiment_output_dir, bucket, copy_commands, file_name="eval_results.json")
        """
        eval_path = os.path.join(experiment_output_dir, 'eval_results.json')
        if not os.path.exists(experiment_output_dir):
            os.makedirs(experiment_output_dir)
        if not os.path.exists(eval_path):
            copy_commands.append(f"gsutil cp {bucket}/{eval_path} {experiment_output_dir}/eval_results.json")
            print(copy_commands)
        """

  run_in_parallel(copy_commands)







names_to_short_names={
"rte_val_acc": "rte (acc)",
"sst2_val_acc": "sst2 (acc)",
"mrpc_val_f1": "mrpc (f1)",
"mrpc_val_acc":"mrpc (acc)",
"stsb_val_pearson_corrcoef": "stsb (pearson)",
"stsb_val_spearman_corrcoef": "stsb (spearman)",
"qqp_val_f1": "qqp (f1)",
"qqp_val_acc": "qqp(acc)",
"mnli_val_acc": "mnli(acc)",
"qnli_val_acc": "qnli(acc)",
"wnli_val_acc": "wnli (acc)",
"cola_val_mcc": "cola (mcc)",
"snli_val_acc": "snli (acc)",
"scitail_val_acc": "scitail (acc)",
"boolq_val_acc": "boolq (acc)",
"mrpc_val_f1": "mrpc (f1)",
"mrpc_val_acc": "mrpc (acc)",
"trec_val_acc": "trec (acc)",
"yelp_polarity_val_acc": "yelp (acc)",
"qqp_val_f1" : "qqp (f1)",
"qqp_val_acc": "qqp (acc)",
"social_i_qa_val_acc": "social_iqa (acc)",
"cosmos_qa_val_acc": "cosmosqa (acc)",
"winogrande_val_acc": "wino (acc)",
"hellaswag_val_acc": "hswag (acc)",
"commonsense_qa_val_acc": "csqa (acc)",

"rte_test_acc": "rte (acc)",
"sst2_test_acc": "sst2 (acc)",
"mrpc_test_f1": "mrpc (f1)",
"mrpc_test_acc":"mrpc (acc)",
"stsb_test_pearson_corrcoef": "stsb (pearson)",
"stsb_test_spearman_corrcoef": "stsb (spearman)",
"qqp_test_f1": "qqp (f1)",
"qqp_test_acc": "qqp(acc)",
"mnli_test_acc": "mnli(acc)",
"qnli_test_acc": "qnli(acc)",
"wnli_test_acc": "wnli (acc)",
"cola_test_mcc": "cola (mcc)",
"snli_test_acc": "snli (acc)",
"scitail_test_acc": "scitail (acc)",
"boolq_test_acc": "boolq (acc)",
"mrpc_test_f1": "mrpc (f1)",
"mrpc_test_acc": "mrpc (acc)",
"trec_test_acc": "trec (acc)",
"yelp_polarity_test_acc": "yelp (acc)",
"qqp_test_f1" : "qqp (f1)",
"qqp_test_acc": "qqp (acc)",
"social_i_qa_test_acc": "social_iqa (acc)",
"cosmos_qa_test_acc": "cosmosqa (acc)",
"winogrande_test_acc": "wino (acc)",
"hellaswag_test_acc": "hswag (acc)",
"commonsense_qa_test_acc": "csqa (acc)",
"unfreeze_lm_head": "lm" ,
"unfreeze_layer_norms": "ln",
"reduction_factor": "r",
"conditional_layer_norm": "cln",
"add_layer_norm_before_adapter": "ln-b",
"add_layer_norm_after_adapter": "ln-a"
}

names = ["snli_{0}_acc", "scitail_{0}_acc", "boolq_{0}_acc", "mrpc_{0}_f1",
         "mrpc_{0}_acc", "trec_{0}_acc", "yelp_polarity_{0}_acc",
         "qqp_{0}_f1", "qqp_{0}_acc", "social_i_qa_{0}_acc", "cosmos_qa_{0}_acc",
         "winogrande_{0}_acc", "hellaswag_{0}_acc", "commonsense_qa_{0}_acc",
         'cola_{0}_mcc', "qnli_{0}_acc", "rte_{0}_acc", "sst2_{0}_acc",
         "stsb_{0}_pearson_corrcoef", "stsb_{0}_spearman_corrcoef",
         "mnli_eval_acc", "wnli_{0}_acc"]

def multiply_accs_by_100 (table):
    for k in names_to_short_names.keys():
        if k in table.columns and "acc" in k:
            table[k] = table[k].apply(lambda x: x*100)

def retrieve_results(output_dir, sweep, short_keys, job_prefix, params=[], ordered_name=False, filename="val_results.json"):
  acc_cols = []
  for acc_col in names:
        metrics = [acc_col.format(name) for name in ["val", "eval", "test"]]
        acc_cols = acc_cols + metrics
  print(job_prefix)
  df = pd.DataFrame()
  keys = flatten(list(sweep.keys()))
  values = list(sweep.values())
  options = [flatten(option) for option in list(itertools.product(*values))]
  i = 0
  for option in options:
    if ordered_name:
        name = job_prefix + "-" + str(i)
        i = i + 1
    else:
        name = make_name(job_prefix, short_keys, option)
    experiment_output_dir = os.path.join(output_dir, name)
    if ordered_name:
        eval_path = os.path.join(experiment_output_dir, filename)
    else:
        eval_path = os.path.join(experiment_output_dir, 'eval_results.json')
    try:
      with open(eval_path, "r") as infile:
        results = json.loads(infile.read())
      results.update({key: value for key, value in zip(keys, option)})
      df = df.append(results, ignore_index=True)
    except FileNotFoundError:
      pass #print("File not found ", eval_path)

  acc_cols_names = list(set.intersection(set(df.columns.tolist()), set(acc_cols)))
  acc_cols_names = [a for a in acc_cols_names if a != "wnli_val_acc" and a != "wnli_test_acc"]
  df = df[params+acc_cols_names]
  if len(params) != 0:
    df = df.sort_values(by=params)

  multiply_accs_by_100(df)

  if "wnli_val_acc" in df.columns:
    df = df.drop(columns="wnli_val_acc")
  if "wnli_test_acc" in df.columns:
    df = df.drop(columns="wnli_test_acc")

  #print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
  # computing the maximum.
  dfs = []
  results_dict = {}
  for acc_col in acc_cols_names:
     params_max = [p  for p in params if p !="learning_rate"]
     if len(params_max) == 0:
        df1 = df.loc[df[acc_col].idxmax()][params_max+[acc_col]]
        results_dict[acc_col]=[df1[acc_col]]
     else:
        df1 = df.loc[df.groupby(params_max)[acc_col].idxmax()][params_max+[acc_col]]
        if len(params_max) != 0:
           df1 = df1.sort_values(by=params_max)
        #print(tabulate(df1, headers='keys', tablefmt='pipe', showindex=False))
        dfs.append(df1)

  if len(params_max) != 0:
    table = dfs[0]
    for i in range(1, len(dfs)):
        right = dfs[i]
        table = pd.merge(table, right, on=params_max)
    acc_cols_names.sort()
    table['average'] = table[acc_cols_names].mean(numeric_only=True, axis=1)
    table = table[params_max+['average']+acc_cols_names]
  else:
    acc_cols_names.sort()
    print(acc_cols_names)
    table = pd.DataFrame.from_dict(results_dict)
    table['average'] = table.mean(numeric_only=True, axis=1)
    #table = table[acc_cols_names+params_max]
    table = table[params_max+['average']+acc_cols_names]
    #table = table.sort_values(by=acc_cols_names)
  table = table.rename(columns=names_to_short_names)
  print(tabulate(table, headers='keys', tablefmt='pipe', showindex=False))
