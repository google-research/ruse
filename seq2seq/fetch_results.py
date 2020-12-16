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
