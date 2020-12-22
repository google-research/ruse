import torch 
import datasets 
dataset = datasets.load_dataset('glue', 'cola', split="train")
print(dataset)

seed = 42
epoch = 1
g = torch.Generator()
g.manual_seed(seed + epoch)
indices = torch.randperm(len(dataset), generator=g).tolist()  # type: ignore

#shuffled_indices = shuffle(range(len(dataset)))
shuffled_dataset = dataset.select(indices)
print(len(shuffled_dataset))


shuffled_dataset = dataset.select([1, 2, 3, 4])
print(shuffled_dataset)
print(len(shuffled_dataset))

