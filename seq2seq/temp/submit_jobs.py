import itertools
import os
from os.path import join

job_string_base = '''
#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l q_short_gpu  -l h=vgn[ehgf]* -P nlu 

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

def transfer():
    for i in range(40):
        job=job_string_base+'python finetune_t5_trainer.py temp_configs/transfer-adapter-{0}.json'.format(i)
        submit_job(job, "adapt"+str(i), outpath_base)

    for i in range(20):
        job=job_string_base+'python finetune_t5_trainer.py temp_configs/transfer-fintune-{0}.json'.format(i)
        submit_job(job, "finetune"+str(i), outpath_base)
        #print(job_string)


transfer()
