from seq2seq.data import AutoTask 


a = AutoTask().get('winogrande').get_dataset(split="train", n_obs=10, add_prefix=True)
for aa in a:
  print(aa)
  break

a = AutoTask().get('cosmos_qa').get_dataset(split="train", n_obs=10, add_prefix=True)
for aa in a:
  print(aa)
  break
