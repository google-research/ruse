import collections
from seq2seq.utils_launch import do_sweep, download_all_evals, retrieve_results

def get_results(sweep, params=None):
    if params is None:
        params = [k for k, v in sweep.items() if len(v) >= 2]
    download_all_evals(sweep, job_prefix, [], sweep["output_dir"][0], ordered_name=True)
    retrieve_results(sweep["output_dir"][0], sweep, [], job_prefix, params, ordered_name=True,
                         filename='val_results.json')
    retrieve_results(sweep["output_dir"][0], sweep, [], job_prefix, params, ordered_name=True,
                         filename='test_results.json')


FUSION_TASKS=["imdb", "snli", "scitail", "boolq", "mrpc", "trec", "yelp_polarity", "qqp", "social_i_qa", "cosmos_qa", "winogrande", "hellaswag", "commonsense_qa"]
GLUE_TASKS=["rte", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "cola"] # T5 does not train on wnli
TRANSFER_TASKS =["imdb", "snli", "scitail", "boolq",  "trec", "yelp_polarity",  "social_i_qa", "cosmos_qa", "winogrande", "hellaswag", "commonsense_qa"]

## adapters
basic_config_path = "configs/paper_experiments/config.json"
job_prefix= "transfer-adapter"
#job_prefix= "fusion-adapters"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 # transfer parameters.
                                 "tasks": [TRANSFER_TASKS],
                                 "eval_tasks": [TRANSFER_TASKS],
                                 "eval_steps": [500],
                                 "model_name_or_path": ["glue-adapters-36"],
                                 "tokenizer_name": ["glue-adapters-36"],
                                 "unfreeze_layer_norms": [True],
                                 "train_task_embeddings": [True],
                                 "optimize_from_scratch": [True],
                                 "freeze_model_but_task_embeddings": [True],
                                 "parametric_task_embedding": [True],
                                 "max_steps": [21845], # just to speed up
                                 "n_train": [100, 500, 1000, 2000], 
                                 # normal parameters.
                                 "load_best_model_at_end": [True], 
                                 "metric_for_best_model": ["average_metrics"], 
                                 "greater_is_better": [True],
                                 "evaluation_strategy": ["steps"],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 # for training adapters.
                                 "train_adapters": [True],                                                         
                                 "reduction_factor": [16], #[8, 16],
                                 #"hidden_dim": [64, 128, 256],
                                 "projected_task_embedding_dim": [64],
                                 #"add_adapters_in_decoder": [True, False],
                                 "conditional_layer_norm": [True], #, False],
                                 "unfreeze_lm_head": [True, False],
                                 "adapter_config_name": ["meta-adapter"],
                                 "train_task_embeddings": [True], #, False],
                                 "non_linearity": ["gelu_new"],
                                 "output_dir": ["outputs/paper_results/test/{0}/transfers/adapters/".format("glue" if "glue" in job_prefix else "fusion")]})
#do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
#get_results(sweep)


## finetune
job_prefix="transfer-fintune"
#job_prefix = "fusion-finetune"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 # transfer parameters.
                                 "tasks": [TRANSFER_TASKS],
                                 "eval_tasks": [TRANSFER_TASKS],
                                 "model_name_or_path": ["glue-finetune-3"],
                                 "tokenizer_name": ["glue-finetune-3"],
                                 "eval_steps": [500],
                                 "max_steps": [21845], # just to speed up 
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 "n_train": [100, 500, 1000, 2000], 
                                 "optimize_from_scratch": [True],
                                 "max_steps": [21845], # just to speed up 
                                 # normal parameters.
                                 "load_best_model_at_end": [True], 
                                 "metric_for_best_model": ["average_metrics"], 
                                 "greater_is_better": [True],
                                 "evaluation_strategy": ["steps"],
                                 "output_dir": ["outputs/paper_results/test/{0}/transfer/finetune/".format("glue" if "glue" in job_prefix else "fusion")]})
#do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
#get_results(sweep)



# trains the models on trained task embeddings.
job_prefix = "trained-task-adapter"
basic_config_path = "configs/experiments/transfer/train_adapter.json"
sweep = collections.OrderedDict({"learning_rate": [3e-4]})
do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)

job_prefix="trained-task-finetune"
basic_config_path = "configs/experiments/transfer/train_finetune.json"
sweep = collections.OrderedDict({"learning_rate": [3e-4]})
do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
