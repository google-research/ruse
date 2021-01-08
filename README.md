# T5 and Multi-task Finetuning

Project realized by **Rabeeh Karimi Mahabadi**([EPFL/Idiap Research Institute](https://www.epfl.ch/schools/ic/)) while interning at Google Brain,
under the supervision of **George Dahl** (Brain) and **Jamie Kiros** (Brain).
*[rabeeh.k68@gmail.com](rabeeh.k68@gmail.com), [gdahl@google.com](gdahl@google.com), [kiros@google.com](kiros@google.com)*

### Description
The objective is to improve the performance of [T5](https://github.com/google-research/text-to-text-transfer-transformer) model on multi-task learning by making use of recently proposed [Adapter layers](https://arxiv.org/abs/1902.00751).
Training effective multi-task learning is a topic of great importance to allow 
capturing shared knowledge from multiple tasks to improve the evaluation performance.

Learning effective multi-task approaches is challenging and requires addressing
several challenges such as catastrophic forgetting, destructive tasks inference, balancing
multiple tasks of various sizes, and over-fitting to low-resource tasks.

### Usage:
To train a model, you can find an example of configuration in the `configs/test.json`
and train the models as follows:

```
python third_party/main/finetune_t5_trainer.py  configs/test.json  

```

### Approaches:
##### Multi-task Fine-tuning with T5:
We extend the HuggingFace T5 model (https://github.com/huggingface/transformers/tree/master/src/transformers/models/t5) to 
handle multiple-tasks, making it allow multi-task finetuning across several datasets. 

##### Multi-task Fine-tuning with Adapter layers: 
We explored the [Adapter layers](https://arxiv.org/abs/1902.00751) idea for multi-task learning. 
Adapters are alternative paradigm of finetuning, introducing a small number of additional 
parameters per task, while keeping the underlying model parameters freezed and shared across all
the tasks while finetuning. This allows them to be inserted into the network with no additional
changes to the structure or parameters. When finetuning, only the adapter layer and layer
normalization parameters are updated.  

### Parallelism:
- The training script supports large scale parallelism on TPUs and on GPUs through `Pytorch XLA`
 and `torch.distributed` respectively. 
- The full data pipeline uses `datasets` library from HuggingFace. This allow to use `shard` operation
to automatically shard the dataset on the TPUs or multiple GPUs. 

### Executables:
- **`third_party/main/finetune_t5_trainer.py`**: Script to launch distributed training of T5
  model on multiple datastes using one of the different approaches.
- **`compute_task_embeddings.py`**: Script to compute the task representation
  from the given datasets by computing the average of sentence embedding over N
  training samples. 
- **`third_party/main/training_args.py`**: Different traninig arguments to specify training and
  models parameters.


### Libraries:
- **`adapters`**: Implements different adapters modules.
- **`data`**: Implements different utilities and processors to convert datasets to sequence to
  sequence format, and a multi-task sampler which ensures the same dataset is
  sampled across TPU cores, this is because the model is adapted per dataset and
  since we train on multiple datasets simultaneously, during backward path, the reduced updates 
  across the TPU cores needs to belong to the same model to allow proper model
  updates in a distributed setting. 
- **`metrics`**: Implements different evaluation metrics to evaluate the
  performance on various datasets' types.
- **`models`**: Implements difference strategies to learn fixed length sentence
  representations from the T5 encoder.
- **`third_party`**: Includes our modifications to HuggingFace T5 model and
  trainer class which allows incorporating adapters in T5 model and expand the
  model to handle training of multiple datasets.
- **`utils`**: Implements different utilities used in the code. 


**This is not an officially supported Google product.**
