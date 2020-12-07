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

outpath_base = "/idiap/temp/rkarimi/internship/sampled/"
if not os.path.exists(outpath_base):
    os.makedirs(outpath_base)
def run_experiment():
    config_path="/idiap/user/rkarimi/dev/ruse/seq2seq/configs/meta_adapter_experiments/train_num_samples_gpu"


    configs = [
       "finetune-100.json",
       "finetune-500.json ",
       "finetune-1000.json",
       "finetune-1500.json",
       "finetune-2000.json",
       "finetune-2400.json",
       "paramteric-meta-adapter-100.json",
       "paramteric-meta-adapter-500.json",
       "paramteric-meta-adapter-1000.json",
       "paramteric-meta-adapter-1500.json ",
       "paramteric-meta-adapter-2000.json"
       "paramteric-meta-adapter-2400.json"]
    for config in configs:
      job_string=job_string_base+'''python finetune_t5_trainer.py  {0}'''
      curr_job = job_string.format(os.path.join(config_path, config))
      submit_job(curr_job, "config-"+config, outpath_base)
run_experiment()
