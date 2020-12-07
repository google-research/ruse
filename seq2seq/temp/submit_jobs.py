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

outpath_base = "/idiap/temp/rkarimi/internship"
if not os.path.exists(outpath_base):
    os.makedirs(outpath_base)
def run_experiment():
    config_path="/idiap/user/rkarimi/dev/ruse/seq2seq/configs/meta_adapter_experiments/gpu"
    configs = ["finetune-2e-5.json",
               "finetune-3e-4.json",
               "paramteric-meta-adapter-1e-2.json",
               "paramteric-meta-adapter-3e-2.json",
               "finetune-3e-3.json",
               "finetune-3e-5.json",
               "paramteric-meta-adapter-3e-1.json",
               "paramteric-meta-adapter-3e-3.json"]
    for config in configs:
      job_string=job_string_base+'''python finetune_t5_trainer.py  {0}'''
      curr_job = job_string.format(os.path.join(config_path, config))
      submit_job(curr_job, "config-"+config, outpath_base)
run_experiment()
