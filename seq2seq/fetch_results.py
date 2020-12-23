import collections 
from utils_launch import retrieve_results, download_all_evals

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
"""

"""
# finetuning both models with different number of samples for steps=140000.
# os.system(f"gsutil rsync -r outputs gs://ruse-xcloud-bucket/{output_dir}")
job_prefix = "m1-adp-v"
short_keys = ["lr", "n", "e", "h"]
params= ["unfreeze_lm_head", "n_finetune", "learning_rate"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [8960, 1792, 896, 448, 224]),
                                 "unfreeze_lm_head": [True, False],
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-meta-task-no-relu-lr-3e-02-emb-n-train-100"],
                                 "eval_output_dir": ["outputs/eval-v-load/finetune-adapter/"]})
output_dir = "outputs/eval-v-load/finetune-adapter/" #outputs/finetune-adapter/"
#download_all_evals(sweep, job_prefix, short_keys, output_dir)
retrieve_results(output_dir, sweep, short_keys, job_prefix, params)



job_prefix = "m1-t5-v"
short_keys = ["lr", "n", "e"]
output_dir = "outputs/eval-v-load/finetune-t5/"
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [8960, 1792, 896, 448, 224]),
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "output_dir": ["mix1-finetune-lr-3e-04"],
                                 "eval_output_dir": ["outputs/eval-v-load/finetune-t5/"]})
#download_all_evals(sweep, job_prefix, short_keys, output_dir)
params= ["n_finetune", "learning_rate"]
retrieve_results(output_dir, sweep, short_keys, job_prefix, params)
"""

"""
output_dir= "outputs/mixture1/meta-adapter/task-emb/"
job_prefix = "m1-meta-task-no-relu-lm"
short_keys = ["lr", 'emb']
params= ["learning_rate"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 "unfreeze_lm_head": [True],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"]})
#download_all_evals(sweep, job_prefix, short_keys, output_dir)
retrieve_results(output_dir, sweep, short_keys, job_prefix, params)

output_dir= "outputs/mixture2/meta-adapter/task-emb/"
job_prefix = "m2-meta-task-no-relu-lm"
short_keys = ["lr", 'emb']
params= ["learning_rate"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 "unfreeze_lm_head": [True],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"]})
#download_all_evals(sweep, job_prefix, short_keys, output_dir)
retrieve_results(output_dir, sweep, short_keys, job_prefix, params)
"""


"""
# only training task-embeddings - this is without loading optimizers.
output_dir = "outputs/eval-v/finetune-only-task-embeddings/"
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
#download_all_evals(sweep, job_prefix, short_keys, output_dir)
params= ["n_finetune", "learning_rate"]
retrieve_results(output_dir, sweep, short_keys, job_prefix, params)
"""

"""
# loading t5 optimizers.
output_dir="outputs/eval-v-load/finetune-t5/"
job_prefix = "m1-load-t5-c" #"m1-t5-v" #-p added 
short_keys = ["lr", "n", "e"]
sweep = collections.OrderedDict({'learning_rate': [1e-2], #, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [8960, 1792, 896, 448, 224]),
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "output_dir": ["mix1-finetune-lr-3e-04"],
                                 "eval_output_dir": ["outputs/eval-v-load/finetune-t5/"]})
#download_all_evals(sweep, job_prefix, short_keys, output_dir)
params= ["n_finetune", "learning_rate"]
retrieve_results(output_dir, sweep, short_keys, job_prefix, params)
"""

"""
# we not loading with load we are loading.
# today jobs
# finetuning both models with different number of samples for steps=140000.
job_prefix = "m1-load-v-c" #"m1-adp-v"
short_keys = ["lr", "n", "e"]
sweep = collections.OrderedDict({'learning_rate': [1e-2], #, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [8960, 1792, 896, 448, 224]),
                                 "unfreeze_lm_head": [True, False],
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["m1-meta-task-no-relu-lr-3e-02-emb-n-train-100"],
                                 #"eval_output_dir": ["outputs/eval-v/finetune-adapter/"]})
                                 "eval_output_dir": ["outputs/eval-v-load/finetune-adapter/"]})
download_all_evals(sweep, job_prefix, short_keys, sweep['eval_output_dir'][0])
params= ["n_finetune", "learning_rate"]
retrieve_results(sweep['eval_output_dir'][0], sweep, short_keys, job_prefix, params)
"""

"""
# t5 without load
job_prefix = "m1-t5-v100" #"m1-t5-v" #-p added
short_keys = ["lr", "n", "e"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 ('n_finetune', 'num_train_epochs'): zip([100, 500, 1000, 2000, 4000],
                                                                         [8960, 1792, 896, 448, 224]),
                                 "do_finetune": [True],
                                 "do_train": [False],
                                 "eval_tasks": [["yelp_polarity", "cola", "snli"]],
                                 "output_dir": ["mix1-finetune-lr-3e-04"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-t5/"]})
#download_all_evals(sweep, job_prefix, short_keys, sweep['eval_output_dir'][0])
params= ["n_finetune", "learning_rate"]
retrieve_results(sweep['eval_output_dir'][0], sweep, short_keys, job_prefix, params)




job_prefix = "m1-adp-v100" #"m1-adp-v"
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
#download_all_evals(sweep, job_prefix, short_keys, sweep['eval_output_dir'][0])
params= ["unfreeze_lm_head", "n_finetune", "learning_rate"]
retrieve_results(sweep['eval_output_dir'][0], sweep, short_keys, job_prefix, params)
"""

"""
# results with less number of iterations.
# running ours and t5 for much less steps.
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
#download_all_evals(sweep, job_prefix, short_keys, sweep['eval_output_dir'][0])
params= ["unfreeze_lm_head", "n_finetune", "learning_rate"]
retrieve_results(sweep['eval_output_dir'][0], sweep, short_keys, job_prefix, params)

# t5 without loading.
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
params= ["n_finetune", "learning_rate"]
#download_all_evals(sweep, job_prefix, short_keys, sweep['eval_output_dir'][0])
retrieve_results(sweep['eval_output_dir'][0], sweep, short_keys, job_prefix, params)
"""
"""
# only fine-tune task-embeddings with lm-head.
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
#download_all_evals(sweep, job_prefix, short_keys, sweep['eval_output_dir'][0])
params= ["n_finetune", "learning_rate"]
retrieve_results(sweep['eval_output_dir'][0], sweep, short_keys, job_prefix, params)
"""

"""
output_dir = "outputs/mixture1/parametric-meta-adapter/task-emb/"
job_prefix = "m1-pmeta-task-norelu" #"m1-pmeta-task-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"]})#,
                                                       # "task_embeddings/n-train-1000",
                                                       # "task_embeddings/n-train-2000",
                                                       # "task_embeddings/n-train-all"]})
params = ["learning_rate"]
download_all_evals(sweep, job_prefix, short_keys, output_dir)
#retrieve_results(output_dir, sweep, short_keys, job_prefix, params)

output_dir = "outputs/mixture2/parametric-meta-adapter/task-emb/"
job_prefix = "m2-pmeta-task-norelu" #"m1-pmeta-task-updd"
short_keys = ["lr", 'emb']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"]})#,
                                                       # "task_embeddings/n-train-1000",
                                                       # "task_embeddings/n-train-2000",
                                                       # "task_embeddings/n-train-all"]})
params = ["learning_rate"]
download_all_evals(sweep, job_prefix, short_keys, output_dir)
#retrieve_results(output_dir, sweep, short_keys, job_prefix, params)



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
params = ["learning_rate", "n_finetune"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["eval_output_dir"][0])
retrieve_results(sweep["eval_output_dir"][0], sweep, short_keys, job_prefix, params)
"""

"""
job_prefix = "m1"
short_keys = ["lr", "h", "r", "n"]
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                 "unfreeze_lm_head": [True, False],
                                 "reduction_factor": [2, 4, 8, 16],
                                 "non_linearity": ["relu", "swish", "tanh", "gelu", "sigmoid"],
                                 "save_steps": [1000],
                                 "per_device_train_batch_size": [16],
                                 "task_embedding_dir": ["task_embeddings/n-train-100"],
                                 "output_dir": ["outputs/mixture1/meta-adapter/task-emb/"]})
#download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
params = ["learning_rate", "unfreeze_lm_head", "reduction_factor", "non_linearity"]
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)
"""

"""
# 17 Dec
# can we train task-embeddings with another network and make it work like this? 
job_prefix = "m1-task"
short_keys = ["lr", 'emb', 'r']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                'projected_task_embedding_dim': [64, 128, 512],
                                 "reduction_factor": [8, 16],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"],
                                 "output_dir": ["outputs/mixture1/meta-adapters-projected-task-embedding"]})
#download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
params = ["reduction_factor", "projected_task_embedding_dim", "learning_rate"]
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)


job_prefix = "m1-p-task"
short_keys = ["lr", 'emb', 'r']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                'projected_task_embedding_dim': [64, 128, 512],
                                 "reduction_factor": [8, 16],
                                 'task_embedding_dir': ["task_embeddings/n-train-100"],
                                 "output_dir": ["outputs/mixture1/parametric-meta-adapters-projected-task-embedding"]})
params = ["reduction_factor", "projected_task_embedding_dim", "learning_rate"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)
"""


"""
job_prefix = "m1"
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
                                 "output_dir": ["m1-task-lr-3e-03-emb-128-r-16/"],
                                 "eval_output_dir": ["outputs/eval-v/finetune-meta-adapters-projected-task-embedding/"]})
params = ["unfreeze_lm_head", "freeze_model_but_task_embeddings", "n_finetune", "learning_rate"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["eval_output_dir"][0])
retrieve_results(sweep["eval_output_dir"][0], sweep, short_keys, job_prefix, params)


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
#params = ["unfreeze_lm_head", "freeze_model_but_task_embeddings", "n_finetune", "learning_rate"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["eval_output_dir"][0])
retrieve_results(sweep["eval_output_dir"][0], sweep, short_keys, job_prefix, params)
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
params = ["unfreeze_lm_head", "reduction_factor", "learning_rate"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)
"""

"""
# test the performance with having one task-projector network.
job_prefix = "m1-task"
short_keys = ["lr", 'emb', 'r']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                'projected_task_embedding_dim': [64, 128, 512],
                                 "reduction_factor": [8, 16],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "train_task_embeddings": [True],
                                 "output_dir": ["outputs/mixture1/meta-adapters-projected-task-embedding-one-task-projector-network"]})
params = ["projected_task_embedding_dim", "reduction_factor", "learning_rate"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)


job_prefix = "m1-p-task"
short_keys = ["lr", 'emb', 'r']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                'projected_task_embedding_dim': [64, 128, 512],
                                 "reduction_factor": [8, 16],
                                 "train_task_embeddings": [True],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "output_dir": ["outputs/mixture1/parametric-meta-adapters-projected-task-embedding-one-task-projector-network"]})
params = ["projected_task_embedding_dim", "reduction_factor", "learning_rate"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)
"""

##############################################################
# 20 Dec
##############################################################
# test the performance with having one task-projector network.
"""
print("with layernorm, unique task embedding projector, meta adapter")
job_prefix = "m1n"
short_keys = ["lr", 'emb', 'r']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                'projected_task_embedding_dim': [64, 128, 512],
                                 "reduction_factor": [8, 16],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "train_task_embeddings": [True],
                                 "unfreeze_layer_norms": [True],
                                 "output_dir": ["outputs/mixture1/meta-adapters-projected-task-embedding-one-task-projector-network-layernorm"]})
params = ["learning_rate", "projected_task_embedding_dim", "reduction_factor"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)



print("with layernorm, unique task embedding projector, parametric meta adapter")
job_prefix = "m1-pn"
short_keys = ["lr", 'emb', 'r']
sweep = collections.OrderedDict({'learning_rate': [1e-2, 3e-1, 3e-2, 3e-3, 3e-4],
                                'projected_task_embedding_dim': [64, 128, 512],
                                 "reduction_factor": [8, 16],
                                 "train_task_embeddings": [True],
                                 "unfreeze_layer_norms": [True],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "output_dir": ["outputs/mixture1/parametric-meta-adapters-projected-task-embedding-one-task-projector-network-layernorm"]})
params = ["learning_rate", "projected_task_embedding_dim", "reduction_factor"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)
"""

"""
# evaluate transfer performance
# gsutil ls gs://ruse-xcloud-bucket/outputs/mixture1/meta-adapters-projected-task-embedding-one-task-projector-network-layernorm/m1n-lr-3e-03-emb-64-r-16
print("evaluation of the trained models with layernorm, still with unfreezing later norms, meta-adapter")
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
params = ["learning_rate", "unfreeze_lm_head", "freeze_model_but_task_embeddings", "n_finetune"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["eval_output_dir"][0])
retrieve_results(sweep["eval_output_dir"][0], sweep, short_keys, job_prefix, params)


print("evaluation of the trained models with layernorm, without unfreezing later norms, meta-adapter")
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
params = ["learning_rate", "unfreeze_lm_head", "freeze_model_but_task_embeddings", "n_finetune"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["eval_output_dir"][0])
retrieve_results(sweep["eval_output_dir"][0], sweep, short_keys, job_prefix, params)


print("evaluation of the trained models with layernorm, still with unfreezing later norms, meta-adapter, longer epochs")
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
params = ["learning_rate", "unfreeze_lm_head", "freeze_model_but_task_embeddings", "n_finetune"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["eval_output_dir"][0])
retrieve_results(sweep["eval_output_dir"][0], sweep, short_keys, job_prefix, params)
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
params = ["learning_rate", "unfreeze_lm_head", "n_finetune"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["eval_output_dir"][0])
retrieve_results(sweep["eval_output_dir"][0], sweep, short_keys, job_prefix, params)
"""


# 22 Dec
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
params = ["learning_rate", "unfreeze_lm_head"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)

# Train the current model, making sure all works
# changing learning rate to lower ones.
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
#download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
params = [ "unfreeze_lm_head", "reduction_factor", "learning_rate", "projected_task_embedding_dim"]
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)

basic_config_path = "configs/experiments/mixture1/meta-task-emb.json"
job_prefix = "m1"
short_keys = ["lr", 'r', 'l']
sweep = collections.OrderedDict({'learning_rate': [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "reduction_factor": [8, 16],
                                 "unfreeze_lm_head": [True, False],
                                 'task_embedding_dir': ["test_data/task_embeddings/n-train-100"],
                                 "train_task_embeddings": [False],
                                 "output_dir": ["outputs/mixture1/meta-adapters-task-projector"]})
#download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
params = [ "unfreeze_lm_head", "reduction_factor", "learning_rate"]
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)
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
params = [ "learning_rate", "unfreeze_lm_head", "n_finetune"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["eval_output_dir"][0])
retrieve_results(sweep["eval_output_dir"][0], sweep, short_keys, job_prefix, params)



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
params = [ "learning_rate", "unfreeze_lm_head", "n_finetune"]
#download_all_evals(sweep, job_prefix, short_keys, sweep["eval_output_dir"][0])
retrieve_results(sweep["eval_output_dir"][0], sweep, short_keys, job_prefix, params)
"""

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
params = [ "learning_rate", "unfreeze_lm_head"]
download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)
"""


# Training the models on multiple tasks, we train both on multiple gpus and 1 gpu 
# baseline on 1 gpu 
"""
basic_config_path = "configs/experiments/glue/finetune.json"
job_prefix = "base1"
short_keys = ["lr"]
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "do_eval": [True],
                                 "output_dir": ["outputs/glue/finetune/num-gpus-1-with-eval"]})
#download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
params = [ "learning_rate"]
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)
"""

# baseline on 4 gpus
basic_config_path = "configs/experiments/glue/finetune.json"
job_prefix = "base4"
short_keys = ["lr"]
sweep = collections.OrderedDict({"learning_rate": [3e-2, 3e-3, 3e-4, 2e-5, 3e-5],
                                 "do_eval": [True],
                                 "output_dir": ["outputs/glue/finetune/num-gpus-4-with-eval"]})
download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
params = [ "learning_rate"]
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)

"""
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
#download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
params = [ "learning_rate", "unfreeze_lm_head", "unfreeze_layer_norms", "reduction_factor"]
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)
"""

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
params = [ "learning_rate", "unfreeze_lm_head", "unfreeze_layer_norms", "reduction_factor"]
download_all_evals(sweep, job_prefix, short_keys, sweep["output_dir"][0])
retrieve_results(sweep["output_dir"][0], sweep, short_keys, job_prefix, params)

