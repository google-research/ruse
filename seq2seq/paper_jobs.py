import collections
from utils_launch import do_sweep

FUSION_TASKS=["imdb", "snli", "scitail", "boolq", "mrpc", "trec", "yelp_polarity", "qqp", "social_i_qa", "cosmos_qa", "winogrande", "hellaswag", "commonsense_qa"]
GLUE_TASKS=["rte", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "wnli", "cola"]
###########################
# experiment on GLUE.
###########################
basic_config_path = "configs/paper_experiments/config.json"
## adapters
"""
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
                                 "output_dir": ["outputs/paper_results/test/glue/adapters/"]})
do_sweep(basic_config_path, sweep, job_prefix="glue-adapters", num_gpus=4, ordered_name=True)
"""
"""
## finetune
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "tasks": [GLUE_TASKS],
                                 "eval_tasks": [GLUE_TASKS],
                                 "model_name_or_path": ["t5-small"],
                                 "tokenizer_name": ["t5-small"],
                                 "num_train_epochs": [20],
                                 "per_device_train_batch_size": [64],
                                 "per_device_eval_batch_size": [64],
                                 "output_dir": ["outputs/paper_results/test/glue/finetune/"]})
do_sweep(basic_config_path, sweep, job_prefix="glue-finetune", num_gpus=4, ordered_name=True)
"""

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
do_sweep(basic_config_path, sweep, job_prefix="glue-finetune-tpu", num_gpus=0, ordered_name=True)
"""

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
do_sweep(basic_config_path, sweep, job_prefix="glue-adapters-tpu", num_gpus=0, ordered_name=True)
