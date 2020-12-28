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
###########################
# experiment on GLUE.
###########################
basic_config_path = "configs/paper_experiments/config.json"

## adapters
job_prefix= "glue-adapters"
#job_prefix= "fusion-adapters"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "tasks": ([GLUE_TASKS] if "glue" in job_prefix else [FUSION_TASKS]),
                                 "eval_tasks": ([GLUE_TASKS] if "glue" in job_prefix else [FUSION_TASKS]),
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "num_train_epochs": [5],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 # for training adapters.
                                 "train_adapters": [True],                                                         
                                 "reduction_factor": [8, 16],
                                 "evaluation_strategy": ["no"],
                                 #"hidden_dim": [64, 128, 256],
                                 "projected_task_embedding_dim": [64],
                                 #"add_adapters_in_decoder": [True, False],
                                 "conditional_layer_norm": [True, False],
                                 "unfreeze_lm_head": [True, False],
                                 "adapter_config_name": ["meta-adapter"],
                                 "train_task_embeddings": [True], #, False],
                                 "non_linearity": ["gelu_new"],
                                 "output_dir": ["outputs/paper_results/test/{0}/adapters/".format("glue" if "glue" in job_prefix else "fusion")]})
#do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
#get_results(sweep)


## adapters
job_prefix= "glue-adapters"
#job_prefix= "fusion-adapters"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "tasks": ([GLUE_TASKS] if "glue" in job_prefix else [FUSION_TASKS]),
                                 "eval_tasks": ([GLUE_TASKS] if "glue" in job_prefix else [FUSION_TASKS]),
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 "load_best_model_at_end": [True], 
                                 "metric_for_best_model": ["average_metrics"], 
                                 "greater_is_better": [True],
                                 # for training adapters.
                                 "train_adapters": [True],
                                 "max_steps": [262144],
                                 "reduction_factor": [32, 16, 8, 4],
                                 "evaluation_strategy": ["no"],
                                 #"hidden_dim": [64, 128, 256],
                                 "projected_task_embedding_dim": [64, 32, 128],
                                 #"add_adapters_in_decoder": [True, False],
                                 "conditional_layer_norm": [True], # False],
                                 "unfreeze_lm_head": [True], # False],
                                 "unfreeze_layer_norms": [True],
                                 "adapter_config_name": ["meta-adapter"],
                                 "train_task_embeddings": [True], #, False],
                                 "non_linearity": ["gelu_new"],
                                 "output_dir": ["outputs/paper_results/full_training/{0}/adapters/".format("glue" if "glue" in job_prefix else "fusion")]})
do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
#get_results(sweep)

## finetune
job_prefix="glue-finetune"
#job_prefix = "fusion-finetune"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "tasks": ([GLUE_TASKS] if "glue" in job_prefix else [FUSION_TASKS]),
                                 "eval_tasks": ([GLUE_TASKS] if "glue" in job_prefix else [FUSION_TASKS]),
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "num_train_epochs": [5],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 "evaluation_strategy": ["no"],
                                 "output_dir": ["outputs/paper_results/test/{0}/finetune/".format("glue" if "glue" in job_prefix else "fusion")]})
#do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
#get_results(sweep)

## finetune
job_prefix="glue-finetune"
#job_prefix = "fusion-finetune"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "tasks": ([GLUE_TASKS] if "glue" in job_prefix else [FUSION_TASKS]),
                                 "eval_tasks": ([GLUE_TASKS] if "glue" in job_prefix else [FUSION_TASKS]),
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "max_steps": [262144],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 "evaluation_strategy": ["no"],
                                 "output_dir": ["outputs/paper_results/full_training/{0}/finetune/".format("glue" if "glue" in job_prefix else "fusion")]})
do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
#get_results(sweep)


# the goal of this experiment is to sweep over more parameters for our model and find the best.
#job_prefix= "glue-adapters"
job_prefix= "fusion-adapters"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "tasks": ([GLUE_TASKS] if "glue" in job_prefix else [FUSION_TASKS]),
                                 "eval_tasks": ([GLUE_TASKS] if "glue" in job_prefix else [FUSION_TASKS]),
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "num_train_epochs": [5],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 # for training adapters.
                                 "add_layer_norm_before_adapter":[True, False],
                                 "add_layer_norm_after_adapter": [True, False],
                                 "train_adapters": [True],                                                         
                                 "reduction_factor": [8, 4], #16],
                                 "evaluation_strategy": ["no"],
                                 #"hidden_dim": [64, 128, 256],
                                 "projected_task_embedding_dim": [64],
                                 #"add_adapters_in_decoder": [True, False],
                                 "conditional_layer_norm": [True], #, False],
                                 "unfreeze_lm_head": [True, False],
                                 "adapter_config_name": ["meta-adapter"],
                                 "unfreeze_layer_norms": [True, False],
                                 "train_task_embeddings": [True], #, False],
                                 "non_linearity": ["gelu_new"],
                                 #"output_dir": ["outputs/paper_results/more_test/fusion/adapters/"]})
                                 "output_dir": ["outputs/paper_results/more_test/{0}/adapters/".format("glue" if "glue" in job_prefix else "fusion")]})
#do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
#get_results(sweep, params=['reduction_factor', 'unfreeze_lm_head', 'unfreeze_layer_norms', 'learning_rate', 'add_layer_norm_before_adapter', 'add_layer_norm_after_adapter'])




# goal: test less reductions.
"""
#job_prefix= "glue-adapters"
job_prefix= "fusion-adapters"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "tasks": [FUSION_TASKS], #
                                 #"tasks": [GLUE_TASKS],
                                 "eval_tasks": [FUSION_TASKS], #
                                 #"eval_tasks": [GLUE_TASKS],
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "num_train_epochs": [5],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 # for training adapters.
                                 "train_adapters": [True],                                                         
                                 "reduction_factor": [4, 2],
                                 "evaluation_strategy": ["no"],
                                 #"hidden_dim": [64, 128, 256],
                                 "projected_task_embedding_dim": [64],
                                 #"add_adapters_in_decoder": [True, False],
                                 "conditional_layer_norm": [True], #, False],
                                 "unfreeze_lm_head": [True], #False],
                                 "adapter_config_name": ["meta-adapter"],
                                 "train_task_embeddings": [True], #, False],
                                 "non_linearity": ["gelu_new"],
                                 "output_dir": ["outputs/paper_results/test_reductions_more/fusion/adapters/"]})
                                 #"output_dir": ["outputs/paper_results/test/glue/adapters/"]})
do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
get_results(sweep)
"""

# goal: test if one linear layer is better
#job_prefix= "glue-adapters"
job_prefix= "fusion-adapters-l"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "tasks": [FUSION_TASKS], #
                                 #"tasks": [GLUE_TASKS],
                                 "eval_tasks": [FUSION_TASKS], #
                                 #"eval_tasks": [GLUE_TASKS],
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "num_train_epochs": [5],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 # for training adapters.
                                 "train_adapters": [True],                                                         
                                 "reduction_factor": [8],
                                 "evaluation_strategy": ["no"],
                                 #"hidden_dim": [64, 128, 256],
                                 "projected_task_embedding_dim": [64],
                                 #"add_adapters_in_decoder": [True, False],
                                 "conditional_layer_norm": [True], #, False],
                                 "unfreeze_lm_head": [True], #False],
                                 "adapter_config_name": ["meta-adapter"],
                                 "train_task_embeddings": [True], #, False],
                                 "non_linearity": ["gelu_new"],
                                 "one_layer_adapter_hyper_net_with_linear": [True],
                                 "output_dir": ["outputs/paper_results/test_one_linear_layer/fusion/adapters/"]})
                                 #"output_dir": ["outputs/paper_results/test/glue/adapters/"]})
#do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
#get_results(sweep)

# goal: test if adapters in decoder are needed.
#job_prefix= "glue-adapters"
job_prefix= "fusion-adapters-d"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "tasks": [FUSION_TASKS], #
                                 #"tasks": [GLUE_TASKS],
                                 "eval_tasks": [FUSION_TASKS], #
                                 #"eval_tasks": [GLUE_TASKS],
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "num_train_epochs": [5],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 # for training adapters.
                                 "train_adapters": [True],                                                         
                                 "reduction_factor": [8],
                                 "evaluation_strategy": ["no"],
                                 #"hidden_dim": [64, 128, 256],
                                 "projected_task_embedding_dim": [64],
                                 #"add_adapters_in_decoder": [True, False],
                                 "conditional_layer_norm": [True], #, False],
                                 "unfreeze_lm_head": [True], #False],
                                 "adapter_config_name": ["meta-adapter"],
                                 "train_task_embeddings": [True], #, False],
                                 "non_linearity": ["gelu_new"],
                                 "add_adapters_in_decoder": [False],
                                 "output_dir": ["outputs/paper_results/test_adapter_in_decoder/fusion/adapters/"]})
                                 #"output_dir": ["outputs/paper_results/test/glue/adapters/"]})
#do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
#get_results(sweep)


# test if setting layer norms unfreezed is better.
job_prefix= "fusion-adapters-n"
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "tasks": ([GLUE_TASKS] if "glue" in job_prefix else [FUSION_TASKS]),
                                 "eval_tasks": ([GLUE_TASKS] if "glue" in job_prefix else [FUSION_TASKS]),
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "num_train_epochs": [5],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 # for training adapters.
                                 "add_layer_norm_before_adapter":[True],
                                 "add_layer_norm_after_adapter": [True],
                                 "train_adapters": [True],                                                         
                                 "reduction_factor": [8], #16],
                                 "evaluation_strategy": ["no"],
                                 #"hidden_dim": [64, 128, 256],
                                 "projected_task_embedding_dim": [64],
                                 #"add_adapters_in_decoder": [True, False],
                                 "conditional_layer_norm": [True], #, False],
                                 "unfreeze_lm_head": [True, False],
                                 "adapter_config_name": ["meta-adapter"],
                                 "unfreeze_layer_norms": [True, False],
                                 "train_task_embeddings": [True], #, False],
                                 "non_linearity": ["gelu_new"],
                                 #"output_dir": ["outputs/paper_results/more_test/fusion/adapters/"]})
                                 "output_dir": ["outputs/paper_results/more_test_layer_norm/{0}/adapters/".format("glue" if "glue" in job_prefix else "fusion")]})
#do_sweep(basic_config_path, sweep, job_prefix=job_prefix, num_gpus=4, ordered_name=True)
#get_results(sweep)


# computes the single-task performances on GLUE.



# testing the tpu speed 
"""
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "tasks": [GLUE_TASKS],
                                 "eval_tasks": [GLUE_TASKS],
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "num_train_epochs": [20],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 # TPU parameters.
                                 "tpu_num_cores": [8],
                                 "predict_with_generate": [False],
                                 "do_eval": [False],
                                 "do_test": [False],
                                 "evaluation_strategy": ["no"],
                                 "prediction_loss_only": [True],
                                 "output_dir": ["outputs/paper_results/test/tpu/glue/finetune/"]})
#do_sweep(basic_config_path, sweep, job_prefix="glue-finetune-tpu", num_gpus=0, ordered_name=True)

# testing adapters on tpu.
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "tasks": [GLUE_TASKS],
                                 "eval_tasks": [GLUE_TASKS],
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "num_train_epochs": [20],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 # for training adapters.
                                 "train_adapters": [True],                                                         
                                 "reduction_factor": [8, 16],
                                 "hidden_dim": [64, 128, 256],
                                 "projected_task_embedding_dim": [64],
                                 "add_adapters_in_decoder": [True, False],
                                 "conditional_layer_norm": [True, False],
                                 "unfreeze_lm_head": [True, False],
                                 "adapter_config_name": ["meta-adapter"],
                                 "train_task_embeddings": [True, False],
                                 "non_linearity": ["swish", "gelu"],
                                 # TPU parameters.
                                 "tpu_num_cores": [8],
                                 "predict_with_generate": [False],
                                 "do_eval": [False],
                                 "do_test": [False],
                                 "evaluation_strategy": ["no"],
                                 "prediction_loss_only": [True],
                                 "output_dir": ["outputs/paper_results/test/tpu/glue/adapters/"]})
#do_sweep(basic_config_path, sweep, job_prefix="glue-adapters-tpu", num_gpus=0, ordered_name=True)
"""
