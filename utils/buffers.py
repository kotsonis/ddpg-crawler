# ReplayBuffer and PrioritizedReplayBuffer classes
# adapted from OpenAI : https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import numpy as np
import torch
import random
from utils import segmenttrees
from operator import itemgetter
from collections import namedtuple, deque
from absl import logging
from absl import flags
config = flags.FLAGS
flags.DEFINE_integer(
    name='memory_size',
    default=1000000,
    help='size of replay memory')
flags.DEFINE_integer(
    name='memory_batch_size',
    default=128,
    help='batch size for replay memory samples')
class ReplayBuffer():
    """ Experience Replay Buffer class """
    def __init__(self, 
               **kwargs):
        """Create simple Replay circular buffer as a list"""

        self._buffer = []
        self._maxsize = kwargs.setdefault('memory_size', config.memory_size)
        self._batch_size = kwargs.setdefault('memory_batch_size',config.memory_batch_size)
        self.experience = (self.__dict__.get('experience',
                                        namedtuple("Experience", 
                                                  field_names=["state", "action", "reward", "next_state", "done"])))
        self.device = kwargs.get('device', 'cpu')
        self._next_idx = 0      # next available index for circular buffer is at start
    

    def __len__(self):
        return len(self._buffer)
    def enough_samples(self):
        """Returns true if buffer has more than batch_size samples"""
        return len(self._buffer) > self._batch_size
        
    #def add(self, state, action, reward, next_state, done, gamma):
    def add(self, **kwargs):
        """ Add a new experience to replay buffer """
        data = self.experience(**kwargs)
        if self._next_idx >= len(self._buffer):         
            # when we are still filling the buffer to capacity
            self._buffer.append(data)
        else:
            # overwrite data at index
            self._buffer[self._next_idx] = data         
        #increment buffer index and loop to beginning when needed
        self._next_idx = int((self._next_idx + 1) % self._maxsize)
        
    def _encode_sample(self, idxes):
        "encode batch of experiences indexed by idxes from buffer"
        #res = [self._buffer[idx] for idx in idxes]
        res = list(itemgetter(*idxes)(self._buffer))
        ret_val = zip(*res)
        ret_list = []
        for vals in ret_val:
            d = torch.tensor(vals).float().to(self.device)
            if (d.dim() < 2):
                d = d.unsqueeze(-1)
            ret_list.append(d)
        return tuple(ret_list)


    def sample(self, **kwargs):
        """Sample a random batch of experiences."""
        batch_size = kwargs.get('batch_size', self._batch_size)
        idxes = [random.randint(0, len(self._buffer) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

