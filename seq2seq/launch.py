import os

def run_jobs(configs_dir, config_name, job_name):
    config_path = os.path.join(configs_dir, config_name)
    command="/google/bin/releases/cloud-alphabetcloud-xcloud/xcloud_cli/xcloud_cli.par google/launch_xla.py  -- --config_path {0} --job_name {1} --num_gpus 1".format(config_path, job_name)
    os.system(command)

"""
# submit adapters
configs_dir = "configs/mixtures1/gpu/adapter/"
configs=[
         "paramteric-meta-adapter-1e-2.json",
         "paramteric-meta-adapter-3e-1.json",
         "paramteric-meta-adapter-3e-2.json",
         "paramteric-meta-adapter-3e-3.json",
         "paramteric-meta-adapter-3e-4.json"
         ]
for config in configs:
    run_jobs(configs_dir, config, "mixture1-100-"+config[:-5])

# submit finetune.
configs_dir = "configs/mixtures1/gpu/finetune"
configs=["finetune-2e-5.json",
         "finetune-3e-3.json",
         "finetune-3e-4.json",
         "finetune-3e-5.json"]
for config in configs:
    run_jobs(configs_dir, config, "mixture1-100-"+config[:-5])

# submit adapters
configs_dir = "configs/mixtures2/gpu/adapter/"
configs=[
         "paramteric-meta-adapter-1e-2.json",
         "paramteric-meta-adapter-3e-1.json",
         "paramteric-meta-adapter-3e-2.json",
         "paramteric-meta-adapter-3e-3.json",
         "paramteric-meta-adapter-3e-4.json"]
for config in configs:
    run_jobs(configs_dir, config, "mixture2-100-"+config[:-5])
# submit finetune.
configs_dir = "configs/mixtures2/gpu/finetune"
configs=["finetune-2e-5.json",
         "finetune-3e-3.json",
         "finetune-3e-4.json",
         "finetune-3e-5.json"]
for config in configs:
    run_jobs(configs_dir, config, "mixture2-100-"+config[:-5])
"""

"""
# submit adapters
configs_dir = "configs/mixtures1-task/gpu/adapter_no_task_emb/"
configs=[
         "paramteric-meta-adapter-1e-2.json",
         "paramteric-meta-adapter-3e-1.json",
         "paramteric-meta-adapter-3e-2.json",
         "paramteric-meta-adapter-3e-3.json",
         "paramteric-meta-adapter-3e-4.json"
         ]
for config in configs:
    run_jobs(configs_dir, config, "mixture1-noinit-"+config[:-5])

# submit adapters
configs_dir = "configs/mixtures2-task/gpu/adapter_no_task_emb/"
configs=[
         "paramteric-meta-adapter-1e-2.json",
         "paramteric-meta-adapter-3e-1.json",
         "paramteric-meta-adapter-3e-2.json",
         "paramteric-meta-adapter-3e-3.json",
         "paramteric-meta-adapter-3e-4.json"]
for config in configs:
    run_jobs(configs_dir, config, "mixture2-noinit-"+config[:-5])
"""

# submit adapters
configs_dir = "configs/mixtures1/gpu/adapter/"
configs=[
         "paramteric-meta-adapter-1e-2.json",
         "paramteric-meta-adapter-3e-1.json",
         "paramteric-meta-adapter-3e-2.json",
         "paramteric-meta-adapter-3e-3.json",
         "paramteric-meta-adapter-3e-4.json"
         ]
for config in configs:
    run_jobs(configs_dir, config, "mix1-lmhead-"+config[:-5])

# submit adapters
configs_dir = "configs/mixtures2/gpu/adapter/"
configs=[
         "paramteric-meta-adapter-1e-2.json",
         "paramteric-meta-adapter-3e-1.json",
         "paramteric-meta-adapter-3e-2.json",
         "paramteric-meta-adapter-3e-3.json",
         "paramteric-meta-adapter-3e-4.json"]
for config in configs:
    run_jobs(configs_dir, config, "mix2-lmhead-"+config[:-5])

"""
# setting only lm-head to true.
configs_dir = "configs/mixtures1-only-lm-head/gpu/adapter/"
configs=[
         "paramteric-meta-adapter-1e-2.json",
         "paramteric-meta-adapter-3e-1.json",
         "paramteric-meta-adapter-3e-2.json",
         "paramteric-meta-adapter-3e-3.json",
         "paramteric-meta-adapter-3e-4.json"
         ]
for config in configs:
    run_jobs(configs_dir, config, "mix1-onlyhead-"+config[:-5])

configs_dir = "configs/mixtures2-only-lm-head/gpu/adapter/"
configs=[
         "paramteric-meta-adapter-1e-2.json",
         "paramteric-meta-adapter-3e-1.json",
         "paramteric-meta-adapter-3e-2.json",
         "paramteric-meta-adapter-3e-3.json",
         "paramteric-meta-adapter-3e-4.json"]
for config in configs:
    run_jobs(configs_dir, config, "mix2-onlyhead-"+config[:-5])
"""
