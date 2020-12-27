import collections
from seq2seq.utils_launch import do_sweep, download_all_evals, retrieve_results

def get_results(sweep, params=None):
    if params is None:
        params = [k for k, v in sweep.items() if len(v) >= 2]
    download_all_evals(sweep, job_prefix, [], sweep["output_dir"][0], ordered_name=True)
    retrieve_results(sweep["output_dir"][0], sweep, [], job_prefix, params, ordered_name=True, filename='val_results.json')
    retrieve_results(sweep["output_dir"][0], sweep, [], job_prefix, params, ordered_name=True, filename='test_results.json')


FUSION_TASKS=["imdb", "snli", "scitail", "boolq", "mrpc", "trec", "yelp_polarity", "qqp", "social_i_qa", "cosmos_qa", "winogrande", "hellaswag", "commonsense_qa"]
GLUE_TASKS=["rte", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "cola"] # T5 does not train on wnli


basic_config_path = "configs/paper_experiments/config.json"
job_prefix="glue-finetune-sigle"
#job_prefix = "fusion-finetune-single"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 ("tasks", "eval_tasks"): zip(GLUE_TASKS if "glue" in job_prefix else FUSION_TASKS,
                                                              GLUE_TASKS if "glue" in job_prefix else FUSION_TASKS),
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "num_train_epochs": [5],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 "evaluation_strategy": ["no"],
                                 "output_dir": ["outputs/paper_results/test/{0}/finetune_single_task/".format("glue" if "glue" in job_prefix else "fusion")]})
do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
