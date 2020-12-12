# python similarity_search.py --model_name_or_path ../outputs/test_man1/ --encoder_pooling attentive --tasks "wmt16-ro-en" "wmt16-en-ro"
# python similarity_search.py --model_name_or_path t5-base --encoder_pooling max --tasks "imdb" --n_obs 10
import argparse
import torch
import logging

from seq2seq.models import T5Config
from seq2seq.tasks import TaskCollator
from transformers import AutoTokenizer
from transformers import set_seed


from torch.utils.data.dataloader import DataLoader
from seq2seq.models import POOLING_MAPPING
from seq2seq.tasks import AutoTask
from collections import defaultdict

# TODO: why relative does not work?
from seq2seq.evaluate.indexing import IndexCreate, IndexSearchMultiple, IndexPrintConfusionMatrix
from seq2seq.evaluate.utils import ClassificationCollator,  _setup_devices
from seq2seq.evaluate.trainer import Trainer
from seq2seq.evaluate.utils import get_train_sampler, is_world_process_zero, get_eval_sampler
from transformers.file_utils import is_torch_tpu_available
from seq2seq.models import T5ForConditionalGeneration

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

logger = logging.getLogger(__name__)


def get_dataloader(args, task, tokenizer, split):
    """Returns a dataloader for the given split of the task."""
    TaskDataset = AutoTask.get(task)
    dataset = TaskDataset.get_dataset(split=split, n_obs=args.n_obs)
    category = TaskDataset.task.category
    if category == "classification":
        label_list = TaskDataset.label_list
        data_collator = ClassificationCollator(tokenizer, args, label_list, args.tpu_num_cores)
    else:
        data_collator = TaskCollator(tokenizer, args, args.tpu_num_cores)

    if split == "train":
        sampler = get_train_sampler(dataset)
    else:
        sampler = get_eval_sampler(dataset)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=args.num_workers,
    )
    return dataloader


def get_embeddings(args, dataloader, model):
    """Given a dataloader and a pretrained T5 model, obtains the embeddings."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            for key, value in inputs.items():
                inputs[key] = value.to(args.device)
            embedding = model(**inputs)[4]
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings)
        return embeddings.cpu().numpy().squeeze()


def split_tasks_per_category(tasks):
    category_to_tasks = defaultdict(list)
    for task in tasks:
        category = AutoTask.get(task).task.category
        category_to_tasks[category].append(task)
    return category_to_tasks


def train_batch_size(args):
    args.train_batch_size = args.per_device_batch_size* max(1, args.n_gpus)
    if is_torch_tpu_available():
        train_batch_size = args.train_batch_size * xm.xrt_world_size()
    args.learning_rate = args.learning_rate*max(1, args.n_gpus)


def main():
    parser = argparse.ArgumentParser("Computes the similarity between different languages.")
    parser.add_argument('--dataloader_num_workers', type=int, default=0,
        help="Number of subprocesses to use for data loading (PyTorch only).\
         0 means that the data will be loaded in the main process.")
    parser.add_argument('--per_device_batch_size', type=int, default=128,
        help="Batch size per GPU core/CPU for training.")
    parser.add_argument("--num_workers", type=int, default=0, help="Defines the number of workers for\
        loading the data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Where do you want to store the pretrained models downloaded from s3")
    #parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--max_source_length", type=int, default=128,
        help = "Maximum source sequence length.")
    parser.add_argument("--max_target_length", type=int, default=128,
        help = "Maximum target sequence length. ")
    parser.add_argument("--n_obs", type=int, default=None, help="Number of data samples.")
    #parser.add_argument("--fixed_length_emb", default=True, action="store_true", help="Whether to learn fixed length"
    #    " embeddings or not. Goes into model.config.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
        help="Defines the learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=4,
        help="Defines the number of training epochs for training the classifier.")
    parser.add_argument("--hidden_dim", type=int, default=512,
        help="Defines the number of hidden states of the classifier.")
    parser.add_argument("--encoder_pooling", type=str, default=None,
        help=f"Pooling layer to use in case of learning fixed length embeddings. "
        f"Selected in {sorted(POOLING_MAPPING.keys())}. Goes into model.config.")
    parser.add_argument("--tpu_num_cores", default=0, help="Defines number of cores of TPU. can be 1 or 8.")
    parser.add_argument("--tasks", nargs="+", type=str, default=[], help=f"The list of tasks to consider.")
    args = parser.parse_args()
    args.device, args.n_gpus = _setup_devices()
    train_batch_size(args)

    # TODO: does logger needs to be set for multiple gpus?
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpus)
    set_seed(args.seed)
    # TODO: this needs to work also with a name not a model directory.
    config = T5Config.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    if is_torch_tpu_available():
        # Set an xla_device flag on the model's config.
        # We'll find a more elegant and not need to do this in the future.
        config.xla_device = True

    args.fixed_length_emb = True
    extra_model_params = ("fixed_length_emb", "encoder_pooling")
    for p in extra_model_params:
        if getattr(args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(args, p))
    # TODO: save the tokenizer too.
    tokenizer = AutoTokenizer.from_pretrained("t5-base", cache_dir=args.cache_dir)
    category_to_tasks = split_tasks_per_category(args.tasks)

    # Trains a classifier on top of freezed embeddings for classification tasks.
    if "classification" in category_to_tasks:
        for task in category_to_tasks["classification"]:
            train_dataloader = get_dataloader(args, task, tokenizer, split="train")
            eval_dataloader = get_dataloader(args, task, tokenizer, split="validation")
            args.num_classes = len(AutoTask.get(task).label_list)
            args.input_dim = config.d_model
            trainer = Trainer(args, config, train_dataloader, eval_dataloader)
            trainer.train()
            accuracy = trainer.evaluate()
            if is_world_process_zero():
                logger.warning("###### task {} accuracy {}".format(task, accuracy.item()))

    # Computes the similarity between language embeddings for translation tasks.
    if "translation" in category_to_tasks:
        model =  T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        from_tf=".ckpt" in args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        ).to(args.device)
        embeddings = []
        for task in category_to_tasks["translation"]:
            dataloader = get_dataloader(args, task, tokenizer, split="validation")
            embeddings.append(get_embeddings(args, dataloader,  model))
        all_data = []
        all_index = []
        for embs in  embeddings:
            idx = IndexCreate(embs, 'FlatL2')
            all_data.append(embs)
            all_index.append(idx)
        err = IndexSearchMultiple(all_data, all_index)
        IndexPrintConfusionMatrix(err, category_to_tasks["translation"])


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
