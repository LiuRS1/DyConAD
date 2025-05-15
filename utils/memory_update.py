from torch import nn
import torch
import math
import warnings
import numbers
from typing import List, Tuple, Optional, overload, Union, cast
import numpy as np

import torch
from torch import Tensor


class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        pass


class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.message_dimension = message_dimension
        self.device = device

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return
        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"

        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update[unique_node_ids] = timestamps
        updated_memory = self.memory_updater(unique_messages, memory)

        self.memory.set_memory(unique_node_ids, updated_memory)

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"

        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps

        return updated_memory, updated_last_update


class EnhancedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(EnhancedGRU, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRUCell(input_size=input_size,
                              hidden_size=hidden_size)
        self.fc = nn.Linear(256, 64)

        self.Linear = nn.Linear(64, hidden_size)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        outputs = self.rnn(input, hx)

        predictions = self.fc(outputs)
        predictions1 = self.Linear(predictions)

        return predictions1.detach()


class EnhancedGRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(EnhancedGRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = EnhancedGRU(message_dimension, memory_dimension)


class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

        self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "egru":
        return EnhancedGRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    else:
        AssertionError('memory updater type is wrong!')






