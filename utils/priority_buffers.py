import numpy as np
import torch
import random
from utils import segmenttrees
from operator import itemgetter
from collections import namedtuple, deque
from absl import logging
from absl import flags
from .buffers import ReplayBuffer

config = flags.FLAGS
flags.DEFINE_float(
    name='PER_alpha',
    default = 0.5,
    help='α factor (prioritization) for Prioritized Replay Buffer')
flags.DEFINE_float(
    name='PER_beta_min',
    default = 0.5,
    help='starting β factor (randomness) for Prioritized Replay Buffer')
flags.DEFINE_float(
    name='PER_beta_max',
    default=1.0,
    help='ending β factor (randomness) for Prioritized Replay Buffer')
flags.DEFINE_float(
    name='PER_minimum_priority',
    default=1e-5,
    help='minimum priority to set when updating priorities')

class PrioritizedReplayBuffer(ReplayBuffer):
    """ A Prioritized according to TD Error replay buffer """
    def __init__(self, **kwargs):
        """Create Prioritized(alpha=0 -> no priority) Replay circular buffer as a list"""
        super(PrioritizedReplayBuffer, self).__init__(**kwargs)
        self._alpha = kwargs.setdefault('PER_alpha', config.PER_alpha)
        assert self._alpha >= 0, "negative alpha not allowed"
        self.beta_min = kwargs.setdefault('PER_beta_min', config.PER_beta_min)
        self.beta_max = kwargs.setdefault('PER_beta_max', config.PER_beta_max)
        self.min_priority = kwargs.get('PER_minimum_priority',config.PER_minimum_priority)
        self._beta_decay = 1
        self._beta = self.beta_min

        # find minimum power of 2 size for segment trees
        st_capacity = 1
        while st_capacity < self._maxsize:
            st_capacity *= 2

        self._st_sum = segmenttrees.SumSegmentTree(st_capacity)
        self._st_min = segmenttrees.MinSegmentTree(st_capacity)
        # set priority with which new experiences will be added. 1.0 means they have highest chance of being sampled
        self._max_priority = 1.0
    def compute_beta_decay(self,training_iterations=1):
        """calculates the beta decay factor according to total training iterations.

        Beta is an annealling factor for randomness. In early training we want to focus on prioritized experiences
        while at the end of training, we should be sampling uniformly."""
        self._beta_decay = (self.beta_max-self.beta_min)/training_iterations
        return
    def decay_beta(self):
        """increases beta used in weights calculation """
        self._beta += self._beta_decay
        self._beta = max(self._beta, self.beta_max)
        return

    def add(self, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx                # obtain next available index to store at from the replay buffer parent class
        super().add(**kwargs)        # add to the replay buffer
        self._st_sum[idx] = self._max_priority ** self._alpha   # put it in the sum tree with max priority
        self._st_min[idx] = self._max_priority ** self._alpha   # put it in the min tree with max priority

    def _sample_proportional(self, **kwargs):
        """ sample uniformly within `batch_size` segments """
        batch_size = kwargs.get('batch_size', self._batch_size)
        results = []
        p_total = self._st_sum.sum(0, len(self._buffer) - 1)       # get total sum of priorites in the whole replay buffer
        every_range_len = p_total / batch_size                      # split the total sum of priorities into batch_size segments
        for i in range(batch_size):
            # generate a random cummulative sum of priorites within this segment
            mass = random.random() * every_range_len + i * every_range_len 
            #Find index in the array of sampling probabilities such that sum of previous values is mass
            idx = self._st_sum.find_prefixsum_idx(mass)             
            results.append(idx)
        return results

    def sample(self, **kwargs):
        """ sample a batch of experiences from memory and also returns importance weights and idxes of sampled experiences"""
        batch_size = kwargs.get('batch_size', self._batch_size)
        idxes = self._sample_proportional(batch_size=batch_size)
        weights = []
        # find maximum weight factor, ie. smallest P(i) since we are dividing by this
        p_sum = self._st_sum.sum()
        p_min = self._st_min.min() / p_sum
        max_weight = (p_min * len(self._buffer)) ** (-self._beta)
        
        for idx in idxes:
            p_sample = self._st_sum[idx] / p_sum
            weight_sample = (p_sample * len(self._buffer)) ** (-self._beta) 
            weights.append(weight_sample / max_weight)
        #expand weights dimension from (batch_size,) to (batch_size,1)
        weights_t = torch.tensor(weights).unsqueeze(1).to(self.device)
        
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights_t, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to transitions at the sampled idxes denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        
        for idx, priority in zip(idxes, priorities):
            assert priority >= 0, "priority must be greater than zero"
            priority = max(priority, self.min_priority)
            assert 0 <= idx < len(self._buffer)
            self._st_sum[idx] = priority ** self._alpha     # update value and parent values in sum-tree
            self._st_min[idx] = priority ** self._alpha     # update value and parent values in min-tree

            self._max_priority = max(self._max_priority, priority)
