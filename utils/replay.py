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
flags.DEFINE_integer(name='memory_size',default=1000000,
                     help='size of replay memory')
flags.DEFINE_integer(name='memory_batch_size',default=128,
                     help='batch size for replay memory samples')
flags.DEFINE_integer(name='n_step',default = 3,
                     help='Number of steps to lookahead the returns in replay buffer')

config = flags.FLAGS

class Buffer():
    """ Experience Replay Buffer class """
    def __init__(self, **kwargs):
        """Create simple Replay circular buffer as a list"""
        self._buffer = []
        self._sampling_results = dict()
        self._maxsize = kwargs.pop('memory_size', config.memory_size)
        self._batch_size = kwargs.pop('memory_batch_size',config.memory_batch_size)
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.device = kwargs.get('device', 'cpu')
        self._next_idx = 0      # next available index for circular buffer is at start

    @property
    def states(self):
        return self._sampling_results['states']
    @property
    def actions(self):
        return self._sampling_results['actions']
    @property
    def rewards(self):
        return self._sampling_results['rewards']
    @property
    def next_states(self):
        return self._sampling_results['next_states']
    @property
    def dones(self):
        return self._sampling_results['dones']
    
    def __len__(self):
        return len(self._buffer)
    @property
    def ready(self):
        """Returns true if buffer has more than batch_size samples"""
        return len(self._buffer) > self._batch_size

    def add(self, **kwargs):
        # basic implementation. store in buffer and increment index
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
        "encodes batch of experiences indexed by idxes from buffer and makes them available in properties"
        #res = [self._buffer[idx] for idx in idxes]
        res = list(itemgetter(*idxes)(self._buffer))
        ret_val = zip(*res)
        ret_list = deque()
        for vals in ret_val:
            d = torch.tensor(vals, requires_grad=False,device=self.device).float()
            if (d.dim() < 2):
                d = d.unsqueeze(-1)
            ret_list.append(d)
        for name in (self.experience._fields):
            self._sampling_results[name]= ret_list.popleft() 
        return

    def sample(self, **kwargs):
        """Sample a random batch of experiences."""
        batch_size = kwargs.get('batch_size', self._batch_size)
        idxes = np.random.randint(len(self._buffer) - 1, size=batch_size)
        # was : idxes = [random.randint(0, len(self._buffer) - 1) for _ in range(batch_size)]
        self._encode_sample(idxes)
        return

#
# nstep replay class
#


class NStepReplay(Buffer):
    """replay buffer with n-step returns"""
    def __init__(self, **kwargs):
        """Creates an n-step  replay buffer.

        arguments:
        'n_step'= number of steps to lookahead in calculating returns
        'gamma' = discount factor per step
        """
        # if not defined by child classes, define what an experience is
        config = kwargs['config']
        super(NStepReplay, self).__init__(**kwargs)
        self.experience = namedtuple("Experience", self.experience._fields + tuple("next_gamma"))
        self.n_step = kwargs.pop('n_step', config.n_step)
        self.gamma = kwargs.pop('gamma', config.gamma)
        self.num_agents = kwargs.pop('num_agents')+1
        #initialize a deque for temporary storage
        self.returns = deque(maxlen=self.n_step)
        return
    @property
    def gammas(self):
        return self._sampling_results['next_gamma']

    def add(self, state, action, reward, next_state, done, **kwargs):
        """Adds experience into temporary n_step buffer, and populates priority buffer as necessary."""

        # get batch size (in case we are stacking multiple agent interactions)
        if isinstance(done, (list, tuple, np.ndarray)):
            B = len(done)
        else:
            B = 0
        
        self.returns.append((state,action,reward,next_state,done, *kwargs))
        if (len(self.returns) == self.n_step):
            state_t, action_t, reward_t, gammas = self._calc_back_rewards(batch_sz=B)
            super().add(
                state=state_t, 
                action=action_t, 
                reward=reward_t,
                next_state=next_state,
                done=done,
                next_gamma=gammas,
                **kwargs)
            
        if np.any(done):
            while (len(self.returns) > 0):
                state_t, action_t, reward_t, gammas = self._calc_back_rewards(batch_sz=B)
                super().add(
                    state=state_t, 
                    action=action_t, 
                    reward=reward_t,
                    next_state=next_state,
                    done=done,
                    next_gamma=gammas,
                    **kwargs)
        return

    def _calc_back_rewards(self,batch_sz):
        """calculates discounted returns for n_step and gives back state,action,discounted_returns,next discount."""
        gamma = self.gamma
        state_t, action_t, reward_t, _ = self.returns.popleft()
        cum_reward = np.array(reward_t).reshape(-1,batch_sz)
        for data in self.returns:
            next_rewards = np.array(data[2]).reshape(-1,batch_sz)
            cum_reward = cum_reward+ gamma*next_rewards
            gamma = gamma*self.gamma
        reward_t = cum_reward.reshape(batch_sz,-1)
        gammas = np.ones(np.array(reward_t).shape)*gamma

        return state_t,action_t,reward_t, gammas

#
# prioritized replay class
#
# Prioritized replay buffer configuration options
flags.DEFINE_float(name='PER_alpha',default = 0.5,
                   help='α factor (prioritization) for Prioritized Replay Buffer')
flags.DEFINE_float(name='PER_beta_min',default = 0.5,
                   help='starting β factor (randomness) for Prioritized Replay Buffer')
flags.DEFINE_float(name='PER_beta_max',default=1.0,
                   help='ending β factor (randomness) for Prioritized Replay Buffer')
flags.DEFINE_float(name='PER_minimum_priority',default=1e-5,
                   help='minimum priority to set when updating priorities')

class PriorityReplay(Buffer):
    """Prioritized replay buffer.
    
    Agnostic with regards to underlying experiences being stored/sampled. """
    def __init__(self, **kwargs):
        """Create Prioritized replay buffer.
        
        parameters:
         PER_alpha: 
            prioritization factor       (CLI `--PER_alpha x.xx`)
         PER_beta_min: 
            initial beta factor         (CLI `--PER_beta_min x.xx`)
         PER_beta_max: 
            final beta factor           (CLI `--PER_beta_max x.xx`)
         PER_minimum_priority: 
            minimum priority for updated indexes
                                        (CLI `--PER_minimum_priority x.xx) """
        # read configuration parameters from arguments or defaults
        self._alpha = kwargs.pop('PER_alpha', config.PER_alpha)
        assert self._alpha >= 0, "negative alpha not allowed"
        self.beta_min = kwargs.pop('PER_beta_min', config.PER_beta_min)
        self.beta_max = kwargs.pop('PER_beta_max', config.PER_beta_max)
        self.min_priority = kwargs.pop('PER_minimum_priority',config.PER_minimum_priority)
        # intialize parent
        super(PriorityReplay, self).__init__(**kwargs)
        # find minimum power of 2 size for sumtree and mintree and create them
        st_capacity = 1
        while st_capacity < self._maxsize:
            st_capacity *= 2
        self._st_sum = segmenttrees.SumSegmentTree(st_capacity)
        self._st_min = segmenttrees.MinSegmentTree(st_capacity)
        # initialize internal parameters
        self._beta_decay = 1
        self._beta = self.beta_min
        self._max_priority = 1.0
        self.vf = np.vectorize(self._st_sum.find_prefixsum_idx)
    
    def compute_beta_decay(self,training_iterations=1):
        """calculates the beta decay factor according to total training iterations.

        beta decay = (`PER_beta_max` - `PER_beta_min`)/`training_iterations`
        
        Beta is an annealling factor for randomness. In early training we want to focus on prioritized experiences
        while at the end of training, we should be sampling uniformly."""
        self._beta_decay = (self.beta_max-self.beta_min)/training_iterations

    def decay_beta(self):
        """increases beta used in weights calculation.
        
        beta = max(`PER_beta_max`, beta + (`PER_beta_max` - `PER_beta_min`)/(total training steps to do)) """
        self._beta += self._beta_decay
        self._beta = max(self._beta, self.beta_max)


    def add(self, **kwargs):
        """adds experience tuple into underlying buffer and updates sumtree/mintree entry with initial priority."""
        idx = self._next_idx                # obtain next available index to store at from the replay buffer parent class
        super().add(**kwargs)        # add to the replay buffer
        self._st_sum[idx] = self._max_priority ** self._alpha   # put it in the sum tree with max priority
        self._st_min[idx] = self._max_priority ** self._alpha   # put it in the min tree with max priority

    def _sample_proportional(self, **kwargs):
        """returns list of indexes from uniformly sampling within `batch_size` segments.
        
        args:
            `batch_size`: number of samples to return (available as commandline parameter) """
        batch_size = kwargs.get('batch_size', self._batch_size)
        results = []
        p_total = self._st_sum.sum(0, len(self._buffer) - 1)       # get total sum of priorites in the whole replay buffer
        bin_interval = p_total / batch_size                      # split the total sum of priorities into batch_size segments
        points_in_bin = np.random.random(size=batch_size)*bin_interval
        offset_per_bin = np.arange(0,batch_size)*bin_interval
        mass = offset_per_bin + points_in_bin
        results = self.vf(mass).tolist()
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
        weights_t = torch.tensor(weights, requires_grad=False,device=self.device).unsqueeze(1)
        
        self._encode_sample(idxes)
        self._sampling_results['weights'] = weights_t
        self._sampling_results['idxes'] = idxes
        return
    @property
    def weights(self):
        return self._sampling_results['weights']
    @property
    def idxes(self):
        return self._sampling_results['idxes']

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

#
# nstep prioritized replay. Just a class that inherits both nstep and prioritized buffer
#

class NStepPriorityReplay(NStepReplay, PriorityReplay):
    def __init__(self, **kwargs):
        super(NStepPriorityReplay,self).__init__(**kwargs)
