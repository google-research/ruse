"""Implements Adapter Controller, a module that keeps multiple 
layers of Adapters, and controlls which adapter layer to use"""

import torch.nn as nn
from seq2seq.adapters import Adapter, AdapterConfig


class AdapterController(nn.Module):
    """Implements Adapter controller module."""

    def __init__(self, tasks, model_config):
        super().__init__()
        self.adapters = nn.ModuleDict(dict())
        self.model_config = model_config
        # We convert the tasks to a list the same way this is done in collator.
        self.tasks = tasks
        self.adapters = self.construct_adapters(tasks)
        self.index_to_tasks = {i: v for i, v in enumerate(tasks)}

    def get_task(self, task):
        return self.index_to_tasks[task]

    def construct_adapters(self, tasks):
        """
        Constructs adapter layers and adds them to a dictionary for the given
        tasks.
        :param tasks: A list of string contraining task names.
        """
        for task in tasks:
            # TODO(rabeeh): for now we have a fixed config for all tasks.
            adapter_config = AdapterConfig()
            adapter = Adapter(self.model_config, adapter_config)
            self.adapters[task] = adapter
        return self.adapters

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

    def convert_to_list(self, tasks):
        if isinstance(tasks, list):
            return tasks
        else:
            return [tasks]

    def enable_adapters(self, tasks):
        """
        Given a list of tasks, it unfreezes their corresponding adapter layers.
        :param tasks: Given list of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for param in adapter.parameters():
                param.requires_grad = True

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
        # Enables the adapter layer for the given task.
        self.enable_adapters(task)
        # Disable other adapters.
        other_tasks = [x for x in self.tasks if x != task]
        self.disable_adapters(other_tasks)
        adapter = self.get_adapter(task)
        return adapter(inputs)
