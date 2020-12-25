from seq2seq.data import AutoTask, TASK_MAPPING 

tasks_translation = ['wmt16-ro-en',  'wmt14-hi-en', 'wmt16-en-ro', 'wmt16-ro-en',
    'wmt16-en-cs',
    'iwslt2017-ro-nl',
    'iwslt2017-en-nl',
    'wmt16-en-fi']

tasks = list(TASK_MAPPING.keys())
for task in tasks:
   if task not in tasks_translation:
      print("### task ", task)
      print("####### train")
      train = AutoTask().get(task).get_dataset(split="train", add_prefix=True, split_validation_test=True)
      print("####### validation")
      validation = AutoTask().get(task).get_dataset(split="validation", add_prefix=True, split_validation_test=True)
      print("####### test")
      test = AutoTask().get(task).get_dataset(split="test", add_prefix=True, split_validation_test=True)
      print("validation ", len(validation)) 
      print("test ", len(test)) 
      print("train ", len(train)) 
      #for aa in a:
      #   print(aa)
      #print("labels ", AutoTask().get(task).label_list)
      print("=================")

#def shuffle(indices):
#  return indices 

#dataset = AutoTask().get("snli").get_dataset(split="train", n_obs=1000)
#shuffled_indices = shuffle(range(len(dataset)))
#shuffled_dataset1 = dataset.select(shuffled_indices[:500])
#shuffled_dataset2 = dataset.select(shuffled_indices[500:])
#print(len(shuffled_dataset1), shuffled_dataset1[0])
#print(len(shuffled_dataset2), shuffled_dataset2[0])
#dataset = AutoTask().get("wnli").get_dataset(split="train", n_obs=1000, split_validation_test=True)
#dataset = AutoTask().get("wnli").get_dataset(split="test", n_obs=1000, split_validation_test=True)
#dataset = AutoTask().get("wnli").get_dataset(split="validation", n_obs=1000, split_validation_test=True)
