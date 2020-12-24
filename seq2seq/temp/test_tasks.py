from seq2seq.data import AutoTask 


a = AutoTask().get('wnli').get_dataset(split="train", n_obs=10, add_prefix=True)
print(a)

for aa in a:
  print(aa)
