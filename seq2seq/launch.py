import os

def run_jobs(configs_dir, config_name, job_name):
    config_path = os.path.join(configs_dir, config_name)
    command="/google/bin/releases/cloud-alphabetcloud-xcloud/xcloud_cli/xcloud_cli.par google/launch_xla.py  -- --config_path {0} --job_name {1} --num_gpus 1".format(config_path, job_name)
    os.system(command)


# submit adapters
configs_dir = "configs/mixtures1-task/gpu/adapter/"
configs=[
         "paramteric-meta-adapter-1e-2.json",
         "paramteric-meta-adapter-3e-1.json",
         "paramteric-meta-adapter-3e-2.json",
         "paramteric-meta-adapter-3e-3.json",
         "paramteric-meta-adapter-3e-4.json"
         ]
for config in configs:
    run_jobs(configs_dir, config, "mixture1-"+config[:-5])

"""
# submit finetune.
configs_dir = "configs/mixtures1/gpu/finetune"
configs=["finetune-2e-5.json",
         "finetune-3e-3.json",
         "finetune-3e-4.json",
         "finetune-3e-5.json"]
for config in configs:
    run_jobs(configs_dir, config, "mixture1-"+config[:-5])
"""

# submit adapters
configs_dir = "configs/mixtures2-task/gpu/adapter/"
configs=[
         "paramteric-meta-adapter-1e-2.json",
         "paramteric-meta-adapter-3e-1.json",
         "paramteric-meta-adapter-3e-2.json",
         "paramteric-meta-adapter-3e-3.json",
         "paramteric-meta-adapter-3e-4.json"]
for config in configs:
    run_jobs(configs_dir, config, "mixture2-"+config[:-5])

"""
# submit finetune.
configs_dir = "configs/mixtures2/gpu/finetune"
configs=["finetune-2e-5.json",
         "finetune-3e-3.json",
         "finetune-3e-4.json",
         "finetune-3e-5.json"]
for config in configs:
    run_jobs(configs_dir, config, "mixture2-"+config[:-5])
"""
