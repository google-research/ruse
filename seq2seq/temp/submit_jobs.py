import os

job_string_base = '''
#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l q_short_gpu  -l h=vgn[e]* -P nlu 

source activate internship
'''

def submit_job(curr_job, filename, outpath_base):
    job_name = "template.job"
    with open(job_name, "w") as f:
       f.write(curr_job)
    os.system("qsub -V -N {0} -e {1}.err -o {1}.out template.job".format(filename, os.path.join(outpath_base, filename)))




def run_experiment(configs_dir, configs, outpath_base):
    if not os.path.exists(outpath_base):
       os.makedirs(outpath_base)
    for config in configs:
      job_string=job_string_base+'''python finetune_t5_trainer.py  {0}'''
      curr_job = job_string.format(os.path.join(configs_dir, config))
      submit_job(curr_job, "config-"+config, outpath_base)

"""
outpath_base = "/idiap/temp/rkarimi/internship/mixture1/adapter/"
configs_dir = "configs/mixtures1/gpu/adapter"
configs=["paramteric-meta-adapter-1e-2.json",
         "paramteric-meta-adapter-3e-1.json",
         "paramteric-meta-adapter-3e-2.json",
         "paramteric-meta-adapter-3e-3.json"]
run_experiment(configs_dir, configs, outpath_base)
"""

outpath_base = "/idiap/temp/rkarimi/internship/mixture2/adapter/"
configs_dir = "configs/mixtures2/gpu/adapter"
configs=["paramteric-meta-adapter-1e-2.json",
         "paramteric-meta-adapter-3e-1.json",
         "paramteric-meta-adapter-3e-2.json",
         "paramteric-meta-adapter-3e-3.json"]
run_experiment(configs_dir, configs, outpath_base)


"""
outpath_base = "/idiap/temp/rkarimi/internship/mixture1/finetune/"
configs_dir = "configs/mixtures1/gpu/finetune/"
configs=["finetune-2e-5.json",  
         "finetune-3e-3.json",
  	 "finetune-3e-4.json",
         "finetune-3e-5.json"]
run_experiment(configs_dir, configs, outpath_base)
"""


outpath_base = "/idiap/temp/rkarimi/internship/mixture2/finetune/"
configs_dir = "configs/mixtures2/gpu/finetune/"
configs=["finetune-2e-5.json",  
         "finetune-3e-3.json",
  	 "finetune-3e-4.json",
         "finetune-3e-5.json"]
run_experiment(configs_dir, configs, outpath_base)
