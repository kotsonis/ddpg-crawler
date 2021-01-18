import numpy as np
import torch
import random
from utils import segmenttrees
from operator import itemgetter
from collections import namedtuple, deque
from absl import logging
from absl import flags
from .priority_buffers import PrioritizedReplayBuffer

config = flags.FLAGS
flags.DEFINE_integer(
    name='n_step',
    default = 3,
    help='Number of steps to lookahead the returns in replay buffer')


class nStepPER(PrioritizedReplayBuffer):
    """A Prioritized Replay Buffer with n-step returns"""
    def __init__(self, **kwargs):
        """Creates an n-step priority replay buffer.
        
        arguments:
        'n_step'= number of steps to lookahead in calculating returns
        'gamma' = discount factor per step
        """
        self.experience = namedtuple("Experience", 
                                    field_names=["state", "action", "reward", "next_state", "done","next_gamma"])
        super(nStepPER, self).__init__(**kwargs)
        self.n_step = kwargs.setdefault('n_step', config.n_step)
        self.gamma = kwargs.setdefault('gamma', config.gamma)
        
        # create discounts array
        self.discount = np.array([self.gamma**i for i in range(0,self.n_step)])
        #initialize a deque for temporary storage
        self.returns = deque(maxlen=self.n_step)
        return

    def _add(self, **kwargs):
        """Adds experience to prioritized buffer."""
        super().add(**kwargs)        # add to the replay buffer
    
    def add(self, **kwargs):
        """Adds experience into temporary n_step buffer, and populates priority buffer as necessary."""
        state = kwargs['state']
        action = kwargs['action']
        reward = kwargs['reward']
        next_state = kwargs['next_state']
        done = kwargs['done']

        # get batch size (in case we are stacking multiple agent interactions)
        B = len(done)
        
        self.returns.append((state,action,reward,next_state,done))
        if (len(self.returns) == self.n_step):
            state_t, action_t, reward_t, gammas = self._calc_back_rewards(batch_sz=B)
            super().add(state=state_t, action=action_t, reward=reward_t, next_state=next_state, done=done, next_gamma=gammas)
            
        if np.any(done):
            while (len(self.returns) > 0):
                state_t, action_t, reward_t, gammas = self._calc_back_rewards(batch_sz=B)
                super().add(state=state_t, action=action_t, reward=reward_t, next_state=next_state, done=done, next_gamma=gammas)
        return

    def _calc_back_rewards(self,batch_sz):
        """calculates discounted returns for n_step and gives back state,action,discounted_returns,next discount."""
        gamma = self.gamma
        state_t, action_t, reward_t, _, _ = self.returns.popleft()
        cum_reward = reward_t.reshape(1,batch_sz)
        for data in self.returns:
            next_rewards = np.array(data[2]).reshape(1,batch_sz)
            cum_reward = cum_reward+ gamma*next_rewards
            gamma = gamma*self.gamma
        reward_t = cum_reward.reshape(batch_sz,1)
        gammas = np.ones(np.array(reward_t).shape)*gamma

        return state_t,action_t,reward_t, gammas
