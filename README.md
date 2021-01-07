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

### Usage
To train a model, you can find an example of configuration in the `configs/test.json`
and train the models as follows:

```
python finetune_t5_trainer.py configs/test.json
```

### Approaches
##### Multi-task Fine-tuning with T5
We extend the HuggingFace T5 model (https://github.com/huggingface/transformers/tree/master/src/transformers/models/t5) to 
handle multiple-tasks, making it allow multi-task finetuning across several datasets. 

##### Multi-task Fine-tuning with Adapter layers 
We explored the [Adapter layers](https://arxiv.org/abs/1902.00751) idea for multi-task learning. 
Adapters are alternative paradigm of finetuning, introducing a small number of additional 
parameters per task, while keeping the underlying model parameters freezed and shared across all
the tasks while finetuning. This allows them to be inserted into the network with no additional
changes to the structure or parameters. When finetuning, only the adapter layer and layer
normalization parameters are updated.  

**This is not an officially supported Google product.**
