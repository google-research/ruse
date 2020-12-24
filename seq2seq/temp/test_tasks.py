from seq2seq.data import AutoTask, TASK_MAPPING 


tasks_translation = ['wmt16-ro-en',  'wmt14-hi-en', 'wmt16-en-ro', 'wmt16-ro-en',
    'wmt16-en-cs',
    'iwslt2017-ro-nl',
    'iwslt2017-en-nl',
    'wmt16-en-fi']

tasks = TASK_MAPPING.keys()
for task in tasks:
   if task not in tasks_translation:
      print("### task ", task)
      a = AutoTask().get(task).get_dataset(split="train", n_obs=4, add_prefix=True)
      for aa in a:
         print(aa)
      print("labels ", AutoTask().get(task).label_list)
      print("=================")
