import collections
from utils_launch import do_sweep

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
job_prefix = "m2-pmeta-task" #-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100",
                                                        "task_embeddings/n-train-1000",
                                                        "task_embeddings/n-train-2000",
                                                        "task_embeddings/n-train-all"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

# 16 Dec
"""
basic_config_path = "configs/experiments/mixture2/paramteric-meta-task-emb.json" #meta-task-emb.json"
job_prefix = "m2-pmeta-task-norelu" #"m2-meta-task-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"]}) #,
                                                        #"task_embeddings/n-train-1000",
                                                        #"task_embeddings/n-train-2000",
                                                        #"task_embeddings/n-train-all"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

# 16 Dec
"""
basic_config_path = "configs/experiments/mixture1/paramteric-meta-task-emb.json"
job_prefix = "m1-pmeta-task-norelu" #"m1-pmeta-task-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"]})#,
                                                       # "task_embeddings/n-train-1000",
                                                       # "task_embeddings/n-train-2000",
                                                       # "task_embeddings/n-train-all"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""
"""
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
# only fine-tune task-embeddings.
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-task-on"
short_keys = ["lr", "n", "e"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [8960, 1792, 896, 448, 224]),
                                 "unfreeze_lm_head": [False],
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "parametric_task_embedding": [True],
                                 "freeze_model_but_task_embeddings": [True],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-meta-task-no-relu-lr-3e-02-emb-n-train-100"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-only-task-embeddings/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""

"""
basic_config_path = "configs/experiments/mixture1/finetune.json"
job_prefix = "m1-load-t5-c" #"m1-t5-v" #-p added 
short_keys = ["lr", "n", "e"]
sweep = collections.OrderedDict({'learning_rate': [1e-2], #, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [8960, 1792, 896, 448, 224]),
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "output_dir": ["mix1-finetune-lr-3e-04"],
                                 #"eval_output_dir": ["outputs/eval-v/finetune-t5/"]})
                                 "eval_output_dir": ["outputs/eval-v-load/finetune-t5/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""

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
"""



# we not loading with load we are loading.
"""
# today jobs
# finetuning both models with different number of samples for steps=140000.
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-load-v-c" #"m1-adp-v"
short_keys = ["lr", "n", "e", "h"]
sweep = collections.OrderedDict({'learning_rate': [1e-2], #, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000], #, 2000, 4000],
                                                                         [8960, 1792, 896]), #, 448, 224]),
                                 "unfreeze_lm_head": [True, False],
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-meta-task-no-relu-lr-3e-02-emb-n-train-100"],
                                 #"eval_output_dir": ["outputs/eval-v/finetune-adapter/"]})
                                 "eval_output_dir": ["outputs/eval-v-load/finetune-adapter/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""




# 15 dec 2020
"""
failed_names= ['m1-t5-noload-lr-1e-02-n-4000-e-224', 'm1-adp-noload-lr-3e-04-n-2000-e-448-h-true', 'm1-adp-noload-lr-3e-04-n-1000-e-896-h-false', 'm1-adp-noload-lr-3e-04-n-1000-e-896-h-true', 'm1-adp-noload-lr-3e-03-n-100-e-8960-h-false', 'm1-adp-noload-lr-3e-03-n-100-e-8960-h-true', 'm1-adp-noload-lr-3e-02-n-4000-e-224-h-false', 'm1-adp-noload-lr-3e-02-n-4000-e-224-h-true', 'm1-adp-noload-lr-3e-02-n-2000-e-448-h-false', 'm1-adp-noload-lr-3e-02-n-2000-e-448-h-true', 'm1-adp-noload-lr-3e-02-n-1000-e-896-h-false', 'm1-adp-noload-lr-3e-02-n-1000-e-896-h-true', 'm1-adp-noload-lr-3e-02-n-500-e-1792-h-true', 'm1-adp-noload-lr-3e-02-n-100-e-8960-h-true', 'm1-adp-noload-lr-3e-01-n-4000-e-224-h-false', 'm1-adp-noload-lr-3e-01-n-2000-e-448-h-false', 'm1-adp-noload-lr-3e-01-n-100-e-8960-h-true', 'm1-adp-noload-lr-1e-02-n-4000-e-224-h-false', 'm1-adp-noload-lr-1e-02-n-4000-e-224-h-true', 'm1-adp-noload-lr-1e-02-n-2000-e-448-h-false', 'm1-adp-noload-lr-1e-02-n-2000-e-448-h-true', 'm1-adp-noload-lr-1e-02-n-1000-e-896-h-false', 'm1-adp-noload-lr-1e-02-n-1000-e-896-h-true', 'm1-adp-noload-lr-1e-02-n-500-e-1792-h-false', 'm1-adp-noload-lr-1e-02-n-500-e-1792-h-true', 'm1-adp-noload-lr-1e-02-n-100-e-8960-h-false', 'm1-adp-noload-lr-1e-02-n-100-e-8960-h-true', 'm1-adp-noload-lr-1e-02-n-1000-e-896-h-true', 'm1-adp-noload-lr-1e-02-n-500-e-1792-h-false', 'm1-adp-noload-lr-1e-02-n-500-e-1792-h-true', 'm1-adp-noload-lr-1e-02-n-100-e-8960-h-false', 'm1-adp-noload-lr-1e-02-n-100-e-8960-h-true', 'm1-adp-v-lr-3e-04-n-4000-e-224-h-false', 'm1-t5-noload-lr-3e-04-n-2000-e-448', 'm1-t5-noload-lr-1e-02-n-4000-e-224', 'm1-adp-v-lr-3e-04-n-100-e-8960-h-true', 'm1-t5-v-lr-3e-03-n-2000-e-448', 'm1-adp-v-lr-3e-03-n-1000-e-896-h-false', 'm1-adp-v-lr-3e-02-n-500-e-1792-h-false', 'm1-adp-v-lr-3e-02-n-100-e-8960-h-false', 'm1-adp-v-lr-3e-02-n-100-e-8960-h-true', 'm1-adp-v-lr-1e-02-n-1000-e-896-h-false', 'm1-adp-v-lr-1e-02-n-1000-e-896-h-true', 'm1-adp-v-lr-1e-02-n-500-e-1792-h-false', 'm1-adp-v-lr-1e-02-n-500-e-1792-h-true', 'm1-adp-v-lr-1e-02-n-100-e-8960-h-false', 'm1-adp-v-lr-1e-02-n-100-e-8960-h-true', 'm1-load-v-c-lr-1e-02-n-1000-e-896-h-false', 'm1-load-v-c-lr-1e-02-n-1000-e-896-h-true', 'm1-load-v-c-lr-1e-02-n-500-e-1792-h-false', 'm1-load-v-c-lr-1e-02-n-500-e-1792-h-true', 'm1-load-v-c-lr-1e-02-n-100-e-8960-h-false', 'm1-load-v-c-lr-1e-02-n-100-e-8960-h-true']

print(failed_names)
"""

"""
# I rerun these after bug fixed.
# our model without loading.
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
#job_prefix = "m1-adp-noload"
job_prefix = "m1-adp-v100"
short_keys = ["lr", "n", "e", "h"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [8960, 1792, 896, 448, 224]),
                                 "unfreeze_lm_head": [True, False],
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "save_steps": [1000],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-meta-task-no-relu-lr-3e-02-emb-n-train-100"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-adapter/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir") #, failed_jobs=failed_names)
"""
"""
# t5 without loading.
basic_config_path = "configs/experiments/mixture1/finetune.json"
job_prefix = "m1-t5-v100" 
#job_prefix = "m1-t5-noload" 
#-p added
short_keys = ["lr", "n", "e"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [8960, 1792, 896, 448, 224]),
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "save_steps": [1000],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "output_dir": ["mix1-finetune-lr-3e-04"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-t5/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir") #, failed_jobs=failed_names)
"""

"""
# only fine-tune task-embeddings with lm-head.
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-task-on-lmhead"
short_keys = ["lr", "n", "e"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [8960, 1792, 896, 448, 224]),
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "parametric_task_embedding": [True],
                                 "freeze_model_but_task_embeddings_and_lm_head": [True],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-meta-task-no-relu-lr-3e-02-emb-n-train-100"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-only-task-embeddings-lm-head/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")


# running ours and t5 for much less steps.
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-adp-half"
short_keys = ["lr", "n", "e", "h"]
sweep = collections.OrderedDict({'learning_rate': [1e-2], #, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([1000, 4000], #100, 500, 1000, 2000, 4000],
                                                                         [720,  180]),            #7200, 1440, 720, 360, 180]),
                                 "unfreeze_lm_head": [True, False],
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "save_steps": [1000],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-meta-task-no-relu-lr-3e-02-emb-n-train-100"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-adapter/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir") #, failed_jobs=failed_names)
"""

"""
# t5 without loading.
basic_config_path = "configs/experiments/mixture1/finetune.json"
job_prefix = "m1-t5-half" #-p added
short_keys = ["lr", "n", "e"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [7200, 1440, 720, 360, 180]),
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "save_steps": [1000],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "output_dir": ["mix1-finetune-lr-3e-04"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-t5/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir") #, failed_jobs=failed_names)
"""


"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-adp-half"
short_keys = ["lr", "n", "e", "h"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [7200, 1440, 720, 360, 180]),
                                 "unfreeze_lm_head": [True, False],
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "save_steps": [1000],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-meta-task-no-relu-lr-3e-02-emb-n-train-100"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-adapter/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir") #, failed_jobs=failed_names)
"""

# Dec 16.
"""
# testing only task-embeddings with parameteric version with smaller number of
# steps.
# best modle:gs://ruse-xcloud-bucket/outputs/mixture1/parametric-meta-adapter/task-emb/m1-pmeta-task-norelu-lr-3e-01-emb-n-train-100
basic_config_path = "configs/experiments/mixture1/paramteric-meta-task-emb.json"
job_prefix = "m1-pmeta-task-on"
short_keys = ["lr", "n", "e"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [7200, 1440, 720, 360, 180]),
                                 "unfreeze_lm_head": [False],
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "parametric_task_embedding": [True],
                                 "freeze_model_but_task_embeddings": [True],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-pmeta-task-norelu-lr-3e-01-emb-n-train-100"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-only-task-embeddings/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""

# testing only task-embeddings with parameteric version with smaller number of
# steps.
# best modle:gs://ruse-xcloud-bucket/outputs/mixture1/parametric-meta-adapter/task-emb/m1-pmeta-task-norelu-lr-3e-01-emb-n-train-100
"""
basic_config_path = "configs/experiments/mixture1/paramteric-meta-task-emb.json"
job_prefix = "m1-pmeta"
short_keys = ["lr", "n", "e"]
sweep = collections.ordereddict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [7200, 1440, 720, 360, 180]),
                                 "unfreeze_lm_head": [true, false],
                                 "do_finetune": [true],
                                 "do_train": [false],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-pmeta-task-norelu-lr-3e-01-emb-n-train-100"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-adapters-paramteric/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""

# do the search also on the adapter layers 
"""
failed_jobs = ['m1-lr-3e-03-h-false-r-16-n-gelu', 'm1-lr-3e-03-h-false-r-2-n-sigmoid', 'm1-lr-3e-03-h-true-r-2-n-gelu', 'm1-lr-3e-02-h-false-r-2-n-relu', 'm1-lr-3e-02-h-true-r-4-n-sigmoid', 'm1-lr-3e-02-h-true-r-2-n-sigmoid', 'm1-lr-3e-02-h-true-r-2-n-gelu', 'm1-lr-3e-02-h-true-r-2-n-tanh', 'm1-lr-3e-02-h-true-r-2-n-swish', 'm1-lr-3e-02-h-true-r-2-n-relu', 'm1-lr-3e-01-h-false-r-16-n-swish', 'm1-lr-3e-01-h-false-r-4-n-swish', 'm1-lr-3e-01-h-false-r-2-n-tanh', 'm1-lr-3e-01-h-true-r-4-n-gelu', 'm1-lr-3e-01-h-true-r-2-n-sigmoid', 'm1-lr-3e-01-h-true-r-2-n-relu', 'm1-lr-1e-02-h-false-r-8-n-relu', 'm1-lr-1e-02-h-true-r-16-n-swish', 'm1-lr-3e-01-h-false-r-2-n-sigmoid', 'm1-lr-3e-01-h-true-r-2-n-sigmoid', 'm1-lr-3e-01-h-true-r-2-n-tanh', 'm1-lr-3e-01-h-true-r-2-n-swish', 'm1-lr-3e-01-h-true-r-2-n-relu', 'm1-lr-1e-02-h-false-r-2-n-sigmoid', 'm1-lr-1e-02-h-false-r-2-n-gelu', 'm1-lr-1e-02-h-false-r-2-n-tanh', 'm1-lr-1e-02-h-false-r-2-n-swish', 'm1-lr-1e-02-h-true-r-2-n-swish', 'm1-lr-1e-02-h-true-r-2-n-tanh', 'm1-lr-1e-02-h-true-r-2-n-relu', 'm1-lr-1e-02-h-true-r-2-n-sigmoid', 'm1-lr-1e-02-h-true-r-2-n-gelu', 'm1-adp-lr-1e-02-n-100-e-7200-h-true-r-2-n-sig', 'm1-adp-lr-1e-02-n-100-e-7200-h-true-r-4-n-rel', 'm1-adp-lr-1e-02-n-100-e-7200-h-true-r-4-n-tan', 'm1-adp-lr-1e-02-n-100-e-7200-h-true-r-4-n-swi', 'm1-adp-lr-1e-02-n-100-e-7200-h-true-r-2-n-swi', 'm1-adp-lr-1e-02-n-100-e-7200-h-true-r-2-n-gel']

basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-c"
short_keys = ["lr", "h", "r", "n"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 "unfreeze_lm_head": [True, False],
                                 "reduction_factor": [2, 4, 8, 16],
                                 "non_linearity": ["relu", "swish", "tanh", "gelu", "sigmoid"],
                                 "save_steps": [2000],
                                 "per_device_train_batch_size": [16],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["outputs/mixture1/finetune-adapter-tune-hyper-params-copy/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix) #, failed_jobs=failed_jobs)
"""

# 17 Dec
# can we train task-embeddings with another network and make it work like this? 
"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-task"
short_keys = ["lr", 'emb', 'r']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                'projected_task_embedding_dim': [64, 128, 512],
                                 "reduction_factor": [8, 16],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"],
                                 "train_task_embeddings": [True],
                                 "output_dir": ["outputs/mixture1/meta-adapters-projected-task-embedding"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path = "configs/experiments/mixture1/paramteric-meta-task-emb.json"
job_prefix = "m1-p-task"
short_keys = ["lr", 'emb', 'r']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                'projected_task_embedding_dim': [64, 128, 512],
                                 "reduction_factor": [8, 16],
                                 "train_task_embeddings": [True],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"],
                                 "output_dir": ["outputs/mixture1/parametric-meta-adapters-projected-task-embedding"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

"""
# finetune the best trained models on the new sets.
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1"
short_keys = ["lr", "n", "e","l", "t"]
# gsutil cp outputs/mixture1/meta-adapters-projected-task-embedding/m1-task-lr-3e-03-emb-128-r-16
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [7200, 1440, 720, 360, 180]),
                                 "unfreeze_lm_head": [True, False],
                                 "freeze_model_but_task_embeddings": [True, False],
                                 'projected_task_embedding_dim': [128],
                                 "reduction_factor": [16],
                                 "do_finetune": [True],
                                 "train_task_embeddings": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-task-lr-3e-03-emb-128-r-16/"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-meta-adapters-projected-task-embedding/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")


# gs://ruse-xcloud-bucket/outputs/mixture1/parametric-meta-adapters-projected-task-embedding/m1-p-task-lr-1e-02-emb-128-r-16
basic_config_path = "configs/experiments/mixture1/paramteric-meta-task-emb.json"
job_prefix = "m1p"
short_keys = ["lr", "n", "e","l", "t"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [7200, 1440, 720, 360, 180]),
                                 "unfreeze_lm_head": [True, False],
                                 "freeze_model_but_task_embeddings": [True, False],
                                 'projected_task_embedding_dim': [128],
                                 "reduction_factor": [16],
                                 "do_finetune": [True],
                                 "train_task_embeddings": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-p-task-lr-1e-02-emb-128-r-16"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-parametric-meta-adapters-projected-task-embedding/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""


# Dec 18
# remove adapter layers from decoder
"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1"
short_keys = ["lr", 'r', 'l']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 "reduction_factor": [8, 16],
                                 "unfreeze_lm_head": [True, False],
                                 'projected_task_embedding_dim': [64], #, 128, 512],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"],
                                 "train_task_embeddings": [True],
                                 "add_adapters_in_decoder": [False],
                                 "output_dir": ["outputs/mixture1/meta-adapters-projected-task-embedding-no-decoder-adapter-t4"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

"""
# test the performance with having one task-projector network.
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1-task"
short_keys = ["lr", 'emb', 'r']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                'projected_task_embedding_dim': [64, 128, 512],
                                 "reduction_factor": [8, 16],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "train_task_embeddings": [True],
                                 "output_dir": ["outputs/mixture1/meta-adapters-projected-task-embedding-one-task-projector-network"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path = "configs/experiments/mixture1/paramteric-meta-task-emb.json"
job_prefix = "m1-p-task"
short_keys = ["lr", 'emb', 'r']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                'projected_task_embedding_dim': [64, 128, 512],
                                 "reduction_factor": [8, 16],
                                 "train_task_embeddings": [True],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "output_dir": ["outputs/mixture1/parametric-meta-adapters-projected-task-embedding-one-task-projector-network"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

# test the performance with having one task-projector network.
"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1n"
short_keys = ["lr", 'emb', 'r']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                'projected_task_embedding_dim': [64, 128, 512],
                                 "reduction_factor": [8, 16],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "train_task_embeddings": [True],
                                 "unfreeze_layer_norms": [True],
                                 "output_dir": ["outputs/mixture1/meta-adapters-projected-task-embedding-one-task-projector-network-layernorm"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)


basic_config_path = "configs/experiments/mixture1/paramteric-meta-task-emb.json"
job_prefix = "m1-pn"
short_keys = ["lr", 'emb', 'r']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                'projected_task_embedding_dim': [64, 128, 512],
                                 "reduction_factor": [8, 16],
                                 "train_task_embeddings": [True],
                                 "unfreeze_layer_norms": [True],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "output_dir": ["outputs/mixture1/parametric-meta-adapters-projected-task-embedding-one-task-projector-network-layernorm"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

# evaluate transfer performance
# gsutil ls gs://ruse-xcloud-bucket/outputs/mixture1/meta-adapters-projected-task-embedding-one-task-projector-network-layernorm/m1n-lr-3e-03-emb-64-r-16
"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1n"
short_keys = ['lr', 'n', 'e', 'l', 't'] #["lr", 'emb', 'l', 't']
sweep = collections.OrderedDict({
                                 'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [7200, 1440, 720, 360, 180]),
                                                                         #[1800, 360, 180, 90, 45]),
                                 "unfreeze_lm_head": [True, False],
                                 "freeze_model_but_task_embeddings": [True, False],
                                 'projected_task_embedding_dim': [64],
                                 "reduction_factor": [16],
                                 "unfreeze_layer_norms": [True],
                                 "do_finetune": [True],
                                 "train_task_embeddings": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["test_data/task_embeddings/n-train-100"],
                                 "output_dir": ["m1n-lr-3e-03-emb-64-r-16"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-meta-adapters-projected-task-emb-with-layer-norm-new-t4/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""

"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1no"
short_keys = ['lr', 'n', 'e', 'l', 't'] #["lr", 'emb', 'l', 't']
sweep = collections.OrderedDict({
                                 'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [7200, 1440, 720, 360, 180]),
                                                                         #[1800, 360, 180, 90, 45]),
                                 "unfreeze_lm_head": [True, False],
                                 "freeze_model_but_task_embeddings": [True, False],
                                 'projected_task_embedding_dim': [64],
                                 "reduction_factor": [16],
                                 "do_finetune": [True],
                                 "train_task_embeddings": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["test_data/task_embeddings/n-train-100"],
                                 "output_dir": ["m1n-lr-3e-03-emb-64-r-16"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-meta-adapters-projected-task-emb-with-layer-norm-new-t4-without-unfreezing/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""

"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1n"
short_keys = ['lr', 'n', 'e', 'l', 't'] #["lr", 'emb', 'l', 't']
sweep = collections.OrderedDict({
                                 'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [7200, 1440, 720, 360, 180]),
                                                                         #[1800, 360, 180, 90, 45]),
                                 "unfreeze_lm_head": [True, False],
                                 "freeze_model_but_task_embeddings": [True, False],
                                 'projected_task_embedding_dim': [64],
                                 "reduction_factor": [16],
                                 "unfreeze_layer_norms": [True],
                                 "do_finetune": [True],
                                 "train_task_embeddings": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["test_data/task_embeddings/n-train-100"],
                                 "output_dir": ["m1n-lr-3e-03-emb-64-r-16"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-meta-adapters-projected-task-emb-with-layer-norm-new-t4-long/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""

# 21 Dec
# lets finetune only the layernorms.
"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1n"
short_keys = ['lr', 'n', 'e', 'l'] #["lr", 'emb', 'l', 't']
sweep = collections.OrderedDict({
                                 'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [7200, 1440, 720, 360, 180]),
                                                                         #[1800, 360, 180, 90, 45]),
                                 "unfreeze_lm_head": [True, False],
                                 "freeze_model": [True],
                                 "unfreeze_layer_norms": [True],
                                 'projected_task_embedding_dim': [64],
                                 "reduction_factor": [16],
                                 "do_finetune": [True],
                                 "train_task_embeddings": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["test_data/task_embeddings/n-train-100"],
                                 "output_dir": ["m1n-lr-3e-03-emb-64-r-16"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-meta-adapters-projected-task-emb-only-layer-norms/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""

################################
# 22 Dec 
################################
"""
# finetune layernorms only.
basic_config_path="configs/experiments/mixture1/finetune.json"
job_prefix = "finetune"
short_keys = ["lr", "l"]
sweep = collections.OrderedDict({'learning_rate': [2e-5, 3e-3, 3e-4, 3e-5],
                                 "unfreeze_lm_head": [True, False],
                                 "output_dir": ["finetune-only-layernorm"],
                                 "freeze_model": [True],
                                 "unfreeze_layer_norms": [True]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""

"""
# Train the current model, making sure all works 
# changing learning rate to lower ones.
#    0 |                  8 |          0.03   |                             64 |        0.877884 |           0.905602 |         0.736617 |
# gsutil ls gs://ruse-xcloud-bucket/outputs/mixture1/meta-adapters-task-projector/m1-lr-3e-02-emb-64-r-8-l-false/
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1"
short_keys = ["lr", 'emb', 'r', 'l']
sweep = collections.OrderedDict({'learning_rate': [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                'projected_task_embedding_dim': [64, 128, 256],
                                 "reduction_factor": [8, 16],
                                 "unfreeze_lm_head": [True, False],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "train_task_embeddings": [True],
                                 "output_dir": ["outputs/mixture1/meta-adapters-task-projector"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)

# gsutil ls gs://ruse-xcloud-bucket/outputs/mixture1/meta-adapters-task-projector/m1-lr-3e-02-r-8-l-false
# 0 |                  8 |          0.03   |        0.876053 |           0.9033   |         0.744876 
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1"
short_keys = ["lr", 'r', 'l']
sweep = collections.OrderedDict({'learning_rate': [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "reduction_factor": [8, 16],
                                 "unfreeze_lm_head": [True, False],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "train_task_embeddings": [False],
                                 "output_dir": ["outputs/mixture1/meta-adapters-task-projector"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""
"""
# evaluate the best model of trained two above commands.
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1p"
short_keys = ['lr', 'n', 'e', 'l']
sweep = collections.OrderedDict({
                                 'learning_rate': [3e-2, 3e-3, 3e-4, 3e-5, 2e-5],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [7200, 1440, 720, 360, 180]),
                                 "unfreeze_lm_head": [True, False],
                                 'projected_task_embedding_dim': [64],
                                 "reduction_factor": [8],
                                 "do_finetune": [True],
                                 "train_task_embeddings": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["test_data/task_embeddings/n-train-100"],
                                 "output_dir": ["m1-lr-3e-02-emb-64-r-8-l-false"],
                                 "eval_output_dir": ["outputs/evals/meta-adapter-projected-task-emb/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")


basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1"
short_keys = ['lr', 'n', 'e', 'l']
sweep = collections.OrderedDict({
                                 'learning_rate': [3e-2, 3e-3, 3e-4, 3e-5, 2e-5],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [7200, 1440, 720, 360, 180]),
                                 "unfreeze_lm_head": [True, False],
                                 "reduction_factor": [8],
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["test_data/task_embeddings/n-train-100"],
                                 "output_dir": ["m1-lr-3e-02-r-8-l-false"],
                                 "eval_output_dir": ["outputs/evals/meta-adapter-without-projected-task-emb/"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
"""


# Training our model in distributed fashion to check
"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m"
short_keys = ["lr", 'r', 'l']
sweep = collections.OrderedDict({'learning_rate': [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "reduction_factor": [8], #, 16],
                                 "unfreeze_lm_head": [True, False],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "train_task_embeddings": [False],
                                 "output_dir": ["outputs/mixture1/meta-adapters-task-projector-new_sampler-num-gpus-1"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, num_gpus=1)
"""

"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "mp"
short_keys = ["lr", 'r', 'l']
sweep = collections.OrderedDict({'learning_rate': [3e-2], # 3e-3, 3e-4, 2e-5, 3e-5],
                                 "reduction_factor": [8], #, 16],
                                 "unfreeze_lm_head": [True, False],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "train_task_embeddings": [False],
                                 "output_dir": ["outputs/mixture1/meta-adapters-task-projector-new_sampler-num-gpus-4"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, num_gpus=4)
"""

"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "xla"
short_keys = ["lr", 'r', 'l']
sweep = collections.OrderedDict({'learning_rate': [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "reduction_factor": [8], #, 16],
                                 "unfreeze_lm_head": [True, False],
                                 "tpu_num_cores": [8],
                                 "prediction_loss_only": [True],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "train_task_embeddings": [False],
                                 "do_eval": [False],
                                 "predict_with_generate": [False],
                                 "output_dir": ["outputs/mixture1/meta-adapters-task-projector-new_sampler-tpu"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, num_gpus=0)

# Evaluates the TPU jobs on GPU.
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "xla"
short_keys = ["lr", 'r', 'l']
sweep = collections.OrderedDict({'learning_rate': [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "reduction_factor": [8], #, 16],
                                 "unfreeze_lm_head": [True, False],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "train_task_embeddings": [False],
                                 "do_eval": [True],
                                 "do_train": [False],
                                 "output_dir": ["outputs/mixture1/meta-adapters-task-projector-new_sampler-tpu"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix) 
"""
"""
# Training the models on multiple tasks, we train both on multiple gpus and 1 gpu 
# baseline on 1 gpu 
basic_config_path = "configs/experiments/glue/finetune.json"
job_prefix = "base1"
short_keys = ["lr"]
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "do_eval": [True],
                                 "output_dir": ["outputs/glue/finetune/num-gpus-1-with-eval"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, num_gpus=1)

# baseline on 4 gpus
basic_config_path = "configs/experiments/glue/finetune.json"
job_prefix = "base4"
short_keys = ["lr"]
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "do_eval": [True],
                                 "output_dir": ["outputs/glue/finetune/num-gpus-4-with-eval"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, num_gpus=4)

# our model on 1 gpus 
basic_config_path = "configs/experiments/glue/meta-task-emb.json"
job_prefix = "our1"
short_keys = ["lr", "r", "ln", "l"]
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "reduction_factor": [8, 16],
                                 "unfreeze_layer_norms": [False, True],
                                 "unfreeze_lm_head": [True, False],
                                 "do_eval": [True],
                                 'projected_task_embedding_dim': [64],
                                 "train_task_embeddings": [True],
                                 "output_dir": ["outputs/glue/adapters/num-gpus-1-with-eval"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, num_gpus=1)


# our model on 4 gpus
basic_config_path = "configs/experiments/glue/meta-task-emb.json"
job_prefix = "our4"
short_keys = ["lr", "r", "ln", "l"]
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "reduction_factor": [8, 16],
                                 "unfreeze_layer_norms": [False, True],
                                 "unfreeze_lm_head": [True, False],
                                 "do_eval": [True],
                                 'projected_task_embedding_dim': [64],
                                 "train_task_embeddings": [True],
                                 "output_dir": ["outputs/glue/adapters/num-gpus-4-with-eval"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, num_gpus=4)
"""

# Training the models on multiple tasks, we train both on multiple gpus and 1 gpu for 100 epochs. 
"""
# baseline on 1 gpu 
basic_config_path = "configs/experiments/glue/finetune.json"
job_prefix = "base11"
short_keys = ["lr"]
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "do_eval": [True],
                                 "num_train_epochs": [100],
                                 "output_dir": ["outputs/glue/finetune/num-gpus-1-with-eval-epochs-100"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, num_gpus=1)

# baseline on 4 gpus
basic_config_path = "configs/experiments/glue/finetune.json"
job_prefix = "base41"
short_keys = ["lr"]
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "do_eval": [True],
                                 "num_train_epochs": [100],
                                 "output_dir": ["outputs/glue/finetune/num-gpus-4-with-eval-100"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, num_gpus=4)

# our model on 1 gpus 
basic_config_path = "configs/experiments/glue/meta-task-emb.json"
job_prefix = "our11"
short_keys = ["lr", "r", "ln", "l"]
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "reduction_factor": [8, 16],
                                 "unfreeze_layer_norms": [False, True],
                                 "unfreeze_lm_head": [True, False],
                                 "num_train_epochs": [100],
                                 "do_eval": [True],
                                 'projected_task_embedding_dim': [64],
                                 "train_task_embeddings": [True],
                                 "output_dir": ["outputs/glue/adapters/num-gpus-1-with-eval-100"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, num_gpus=1)



# our model on 4 gpus
basic_config_path = "configs/experiments/glue/meta-task-emb.json"
job_prefix = "our41"
short_keys = ["lr", "r", "ln", "l"]
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "reduction_factor": [8, 16],
                                 "unfreeze_layer_norms": [False, True],
                                 "unfreeze_lm_head": [True, False],
                                 "do_eval": [True],
                                 'projected_task_embedding_dim': [64],
                                 "num_train_epochs": [100],
                                 "train_task_embeddings": [True],
                                 "output_dir": ["outputs/glue/adapters/num-gpus-4-with-eval-100"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, num_gpus=4)

"""

# Tests the conditional layer norm.
"""
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "c"
short_keys = ["lr", 'r', 'l', 't']
sweep = collections.OrderedDict({'learning_rate': [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "reduction_factor": [8, 16],
                                 "unfreeze_lm_head": [True, False],
                                 "train_task_embeddings": [False, True],
                                 "conditional_layer_norm": [True],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "output_dir": ["outputs/mixture1/meta-adapters-task-projector-conditional-layer-norm"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix)
"""


# transfer performance with conditional layer norms.
# gsutil cp gs://ruse-xcloud-bucket/outputs/mixture1/meta-adapters-task-projector-conditional-layer-norm/c-lr-3e-03-r-16-l-false-t-false/
#    16 |                  0 |                       0 |          0.003  |        0.876968 |           0.915579 |         0.741817 |
basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "c"
short_keys = ["lr", 'n', 'e', 'l', 't']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [8960, 1792, 896, 448, 224]),
                                 "unfreeze_lm_head": [False, True],
                                 "train_task_embeddings": [False, True],
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "reduction_factor": [16],
                                 "conditional_layer_norm": [True],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "output_dir": ["c-lr-3e-03-r-16-l-false-t-false/"]})
                                 "eval_output_dir": ["outputs/eval-v/conditional-layer-norm"]})
do_sweep(basic_config_path, sweep, short_keys, job_prefix, output_dir_name="eval_output_dir")
