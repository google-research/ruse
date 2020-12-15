with open("text") as f:
    lines = f.readlines()

job_name_ind = 0
status_ind = 20
failed_jobs = []
for line in lines:
    splits = line.split()
    print(splits[0].rstrip().split('.'))
    job_id = splits[0].rstrip().split('.')[1]
    print(job_id)
    status = splits[-1].rstrip()
    date = splits[2].rstrip()
    if status == "FAILED" and date == "2020-12-15":
        failed_jobs.append(job_id)
print(failed_jobs)
