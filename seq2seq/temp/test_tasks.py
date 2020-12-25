from seq2seq.data import AutoTask, TASK_MAPPING 

"""
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
"""

def shuffle(indices):
  return indices 

dataset = AutoTask().get("snli").get_dataset(split="train", n_obs=1000)
shuffled_indices = shuffle(range(len(dataset)))
shuffled_dataset1 = dataset.select(shuffled_indices[:500])
shuffled_dataset2 = dataset.select(shuffled_indices[500:])
print(len(shuffled_dataset1), shuffled_dataset1[0])
print(len(shuffled_dataset2), shuffled_dataset2[0])
