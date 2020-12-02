import itertools
import os
from os.path import join

job_string_base = '''
#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l q_gpu  -l h=vgn[hgf]* -P nlu 

source activate internship
'''

def submit_job(curr_job, filename, outpath_base):
    job_name = "template.job"
    with open(job_name, "w") as f:
       f.write(curr_job)
    os.system("qsub -V -N {0} -e {1}.err -o {1}.out template.job".format(filename, os.path.join(outpath_base, filename)))



outpath_base = "/idiap/temp/rkarimi/ruse/"
if not os.path.exists(outpath_base):
    os.makedirs(outpath_base)

def run():
     job_string=job_string_base+'''python finetune_t5_trainer.py  configs/joint_adapter_local.json'''
     curr_job = job_string_base+job_string
     submit_job(curr_job, "joint", outpath_base)

     

run()
