import torch
import torch.nn as nn

from .adapter_utils import MetaUpSampler, MetaDownSampler
from .adapter_modeling import MetaAdapter


class MetaAdapterController(nn.Module):
  """Implements Adapter controller module which generates
   the adapter layers embeddings."""

  def __init__(self, config):
    super().__init__()
    self.adapters = nn.ModuleDict(dict())
    self.config = config
    self.tasks = config.tasks
    self.adapters = self.construct_adapters(self.tasks)
    self.task_embedding_dir = config.task_embedding_dir
    self.input_dim = config.input_dim

    """
    self.task_to_embeddings = {}
    for task in self.tasks:
      #  #task_embedding_path=os.path.join(self.task_embedding_dir, task+".npy")
      #  # TODO: device needs to be set properly.
      self.task_to_embeddings[task] = torch.randn(config.task_embedding_dim).cuda()
      #adapter_config.task_embedding_dim).cuda()
      #  #print("### self.task_to_embeddings ", self.task_to_embeddings[task])
      #  #torch.Tensor(np.load(task_embedding_path)).cuda()
    """
    self.task_to_embeddings = nn.ParameterDict({
      task: nn.Parameter(torch.randn((config.task_embedding_dim))) for task in self.tasks})
    self.meta_up_sampler = MetaUpSampler(config)
    self.meta_down_sampler = MetaDownSampler(config)
    self.task_to_adapter = {task: task for task in self.tasks}

  def enable_adapters(self, tasks):
    """
    Given a list of tasks, it unfreezes their corresponding adapter layers.
    :param tasks: Given list of tasks.
    """
    tasks = self.convert_to_list(tasks)
    for task in tasks:
      adapter = self.get_adapter(task)
      for name, param in adapter.named_parameters():
        param.requires_grad = True

  def disable_adapters(self, tasks):
    """
    Given a list of tasks, it freezes their corresponding adapter layers'
    parameters.
    :param tasks: Given list of tasks.
    """
    tasks = self.convert_to_list(tasks)
    for task in tasks:
      adapter = self.get_adapter(task)
      for param in adapter.parameters():
        param.requires_grad = False

  def construct_adapters(self, tasks):
    """
    Constructs adapter layers and adds them to a dictionary for the given
    tasks.
    :param tasks: A list of string contraining task names.
    """
    for task in tasks:
      self.adapters[task] = MetaAdapter(self.config)
    return self.adapters

  def set_task_to_adapter_map(self, mapping):
    self.task_to_adapter = mapping

  def get_task(self, task):
    return self.task_to_adapter[task]


  def convert_to_list(self, tasks):
    if isinstance(tasks, list):
      return tasks
    else:
      return [tasks]

  def get_adapter(self, task):
    """Given a task returns its corresponding adapter layer, returns None
    if task is not registered.
    :param task: Input task name.
    :return: Adapter layer corresponding to the given task.
    """
    return self.adapters[task]

  def forward(self, task, inputs):
    """Retrieves the adapter layer corresponding to the given
    task. It freezes the adapter layers for all the other tasks
    and call the selected adapter layer.
    :param task: the name of the current task.
    :param inputs: the inputs to feed in in the adapter layer.
    :return: outputs of the adapter layer."""
    task = self.get_task(task)
    adapter = self.get_adapter(task)

    # Enables the adapter layer for the given task.
    self.enable_adapters(task)
    # Disable other adapters.
    other_tasks = [x for x in self.tasks if x != task]
    self.disable_adapters(other_tasks)

    # Generates the weights/biases for up and down sampler and sets them.
    # TODO: this is only correct if adapter_config.add_layer_norm_before_adapter is set to True.
    # TODO: remove the layer norm from the down_sampler.
    weight_up, bias_up = self.meta_up_sampler(self.task_to_embeddings[task])
    weight_down, bias_down = self.meta_down_sampler(self.task_to_embeddings[task])
    return adapter(inputs, weight_down, bias_down, weight_up, bias_up)




