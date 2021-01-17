# ReplayBuffer and PrioritizedReplayBuffer classes
# adapted from OpenAI : https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import numpy as np
import torch
import random
from utils import segmenttrees
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
        res = np.array(itemgetter(*idxes)(self._buffer), dtype=object)
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

class PrioritizedReplayBuffer(ReplayBuffer):
    """ A Prioritized according to TD Error replay buffer """
    def __init__(self, **kwargs):
        """Create Prioritized(alpha=0 -> no priority) Replay circular buffer as a list"""
        super(PrioritizedReplayBuffer, self).__init__(**kwargs)
        self._alpha = kwargs.setdefault('PER_alpha', config.PER_alpha)
        assert self._alpha >= 0, "negative alpha not allowed"

        # find minimum power of 2 size for segment trees
        st_capacity = 1
        while st_capacity < self._maxsize:
            st_capacity *= 2

        self._st_sum = segmenttrees.SumSegmentTree(st_capacity)
        self._st_min = segmenttrees.MinSegmentTree(st_capacity)
        # set priority with which new experiences will be added. 1.0 means they have highest chance of being sampled
        self._max_priority = 1.0

    def add(self, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx                # obtain next available index to store at from the replay buffer parent class
        super().add(**kwargs)        # add to the replay buffer
        self._st_sum[idx] = self._max_priority ** self._alpha   # put it in the sum tree with max priority
        self._st_min[idx] = self._max_priority ** self._alpha   # put it in the min tree with max priority

    def _sample_proportional(self, batch_size: int):
        """ sample uniformly within `batch_size` segments """
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

    def sample(self, batch_size:int, beta:float=1):
        """ sample a batch of experiences from memory and also returns importance weights and idxes of sampled experiences"""
        assert beta > 0
        idxes = self._sample_proportional(batch_size)
        weights = []
        # find maximum weight factor, ie. smallest P(i) since we are dividing by this
        p_sum = self._st_sum.sum()
        p_min = self._st_min.min() / p_sum
        max_weight = (p_min * len(self._buffer)) ** (-beta)
        
        for idx in idxes:
            p_sample = self._st_sum[idx] / p_sum
            weight_sample = (p_sample * len(self._buffer)) ** (-beta) 
            weights.append(weight_sample / max_weight)
        #expand weights dimension from (batch_size,) to (batch_size,1)
        weights_t = torch.tensor(weights).unsqueeze(1).to(device)
        
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
            assert priority > 0, "priority must be greater than zero"
            assert 0 <= idx < len(self._buffer)
            self._st_sum[idx] = priority ** self._alpha     # update value and parent values in sum-tree
            self._st_min[idx] = priority ** self._alpha     # update value and parent values in min-tree

            self._max_priority = max(self._max_priority, priority)

class nStepPER(PrioritizedReplayBuffer):
    """A Prioritized Replay Buffer with n-step returns"""
    def __init__(self, size: int, batch_size: int, alpha: float, n_step: float, gamma: float):
        super(nStepPER, self).__init__(size, batch_size, alpha)
        self.gamma = gamma
        self.steps = n_step
        #initialize a deque for temporary storage
        self.returns = deque(maxlen=n_step)
        return

    # override the PER add method
    def add(self, state_N, action_N, reward_N, next_state_N, done_N):
        "add an experience to a n_step PER. This method first fills out an n_step buffer"
        self.returns.append((state_N, action_N, reward_N, next_state_N, done_N))
        gamma = self.gamma
        if (len(self.returns) == self.steps):
            state_t, action_t, reward_t, next_state_t, done_t = self.returns.popleft()
            for data in self.returns:
                next_rewards = np.array(data[2])
                reward_t = reward_t+ gamma*next_rewards
                gamma = gamma*self.gamma
            gammas = np.ones(np.array(reward_t).shape)*gamma
            #super().add(state_t, action_t, reward_t, next_state_N, done_N, gammas)
            for i in range(len(state_t)): 
                super().add(state_t[i],action_t[i],reward_t[i],next_state_N[i],done_N[i], gammas[i])
        if np.any(done_N):
            while (len(self.returns) > 0):
                gamma = self.gamma
                state_t, action_t, reward_t, next_state_t, done_t = self.returns.popleft()
                for data in self.returns:
                    next_rewards = np.array(data[2])
                    reward_t = reward_t+ gamma*next_rewards
                    gamma = gamma*self.gamma
                gammas = np.ones(np.array(reward_t).shape)*gamma
                #super().add(state_t, action_t, reward_t, next_state_N, done_N, gammas)
                for i in range(len(state_t)): 
                    super().add(state_t[i],action_t[i],reward_t[i],next_state_N[i],done_N[i], gammas[i])
        return



