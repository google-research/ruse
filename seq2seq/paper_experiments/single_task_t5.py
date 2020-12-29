import pandas as pd
import collections
from seq2seq.utils_launch import do_sweep, download_all_evals, retrieve_results
from tabulate import tabulate
from seq2seq.utils_launch import order_glue_columns
import math 

def get_results(sweep, params=None, single_task=False, tasks=[]):
    if params is None:
        params = [k for k, v in sweep.items() if len(v) >= 2]
    download_all_evals(sweep, job_prefix, [], sweep["output_dir"][0], ordered_name=True)
    if single_task:
        params = [p for p in params if p !=('tasks', 'eval_tasks')]
        val_tables = []
        test_tables = []
        for task in tasks:
            val_tables.append(retrieve_results(sweep["output_dir"][0], sweep, [], job_prefix, params, ordered_name=True, filename='val_results.json', task=[task]))
            test_tables.append(retrieve_results(sweep["output_dir"][0], sweep, [], job_prefix, params, ordered_name=True, filename='test_results.json', task=[task]))
        val_table = pd.concat(val_tables, axis=1)
        test_table = pd.concat(test_tables, axis=1)
        del val_table['average']
        del test_table['average']
        val_cols = list(val_table.columns)
        test_cols = list(test_table.columns)
        val_table['average'] = val_table.mean(numeric_only=True, axis=1)
        test_table['average'] = test_table.mean(numeric_only=True, axis=1)
        val_table = order_glue_columns(val_table)
        test_table = order_glue_columns(test_table)
        print("val results")
        print(tabulate(val_table, headers='keys', tablefmt='pipe', showindex=False))
        print(tabulate(val_table, headers='keys', tablefmt='latex', showindex=False))

        print("test results")
        print(tabulate(test_table, headers='keys', tablefmt='pipe', showindex=False))
        print(tabulate(test_table, headers='keys', tablefmt='latex', showindex=False))
    else:
        retrieve_results(sweep["output_dir"][0], sweep, [], job_prefix, params, ordered_name=True,
                         filename='val_results.json')
        retrieve_results(sweep["output_dir"][0], sweep, [], job_prefix, params, ordered_name=True,
                         filename='test_results.json')

FUSION_TASKS=["imdb", "snli", "scitail", "boolq", "mrpc", "trec", "yelp_polarity", "qqp", "social_i_qa", "cosmos_qa", "winogrande", "hellaswag", "commonsense_qa"]
GLUE_TASKS=["rte", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "cola"] # T5 does not train on wnli


#sizes = {"rte": 2489, "glue":949090, "cola":8550 , "sst2":67348, "mrpc":3667, "stsb":5748, "qqp":363845, "mnli":392701, "qnli":104742}
def get_single_tasks(tasks):
  results= [([task], [task]) for task in tasks]
  return results



basic_config_path = "configs/paper_experiments/config.json"
job_prefix="glue-finetune-single"
#job_prefix = "fusion-finetune-single"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 ("tasks", "eval_tasks"): get_single_tasks(GLUE_TASKS if "glue" in job_prefix else FUSION_TASKS),
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "num_train_epochs": [5],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 "evaluation_strategy": ["no"],
                                 "output_dir": ["outputs/paper_results/test/{0}/finetune_single_task/".format("glue" if "glue" in job_prefix else "fusion")]})
#do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
#get_results(sweep, single_task=True, tasks=GLUE_TASKS)


job_prefix="glue-finetune-single"
#job_prefix = "fusion-finetune-single"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 ("tasks", "eval_tasks"): get_single_tasks(GLUE_TASKS if "glue" in job_prefix else FUSION_TASKS),
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "load_best_model_at_end": [True], 
                                 "metric_for_best_model": ["average_metrics"], 
                                 "greater_is_better": [True],
                                 "evaluation_strategy": ["steps"],
                                 "max_steps": [65536],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 "output_dir": ["outputs/paper_results/paper_results/{0}/finetune_single_task/".format("glue" if "glue" in job_prefix else "fusion")]})
do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
#get_results(sweep, single_task=True, tasks=GLUE_TASKS)
