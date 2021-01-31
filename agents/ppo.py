""" RL Agent based on Proximal Policy Optimization  """
import numpy as np
from absl import logging
from absl import flags
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.OUNoise import OUNoise
import copy
import networks
import agents.base
from utils import replay

config = flags.FLAGS

class PPO(agents.base.Agent):
    """PPO Agent """
    def __init__(self, **kwargs):
        # use a replay buffer that stores also log probabilities
        kwargs['replay_buffer_class']= replay.PPO_buffer
        # ----------------- create online & target actors -------------------- #
        actor_dnn_class = kwargs.setdefault('actor_dnn_class',networks.PPOActor)
        critic_dnn_class = kwargs.setdefault('critic_dnn_class',networks.PPOCritic)
        # initialize base class
        super(PPO,self).__init__(**kwargs)

    # we need to override the parent train function, since PPO training steps
    # as as follows:
    # REPEAT until max interations done
    #   STEP 1: Store trajectories based on current policy
    #   STEP 2: REPEAT until training steps reached
    #       STEP 1: Sample trajectories from buffer
    #       STEP 2: REPEAT until improvement steps reached
    #               STEP 1: Calculate Q future

    #   STEP 3: REPEAT until must populate
    # a buffer with trajectories based 
    def train(self,**kwargs):
        max_episodes = kwargs.setdefault('max_iterations',config.max_iterations)

        for _ in range(max_episodes):
            states,actions,rewards,next_states,dones,old_probs = self.sample()
        
        
    def capture_episode(self, **kwargs):
        max_frames_per_episode = kwargs.setdefault('max_frames_per_episode', config.max_frames_per_episode)
        # clear up buffer
        self.memory.clear()
        env = self.env
        env_info = env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
        dones = np.zeros((self.num_agents,))
        frames = 0

        while not np.any(dones):
            actions, action_probs = self.actor(states)
            env_info = env.step(actions)[self.brain_name]
            next_states = env_info.vector_observations
            rewards = np.array(env_info.rewards)
            # fix NaN rewards of crawler environment, by penalizing a NaN reward
            rewards = np.nan_to_num(rewards, nan=-5.0)
            dones = env_info.local_done                       # see if episode finished
            frames += 1
            self.memory.add(
                states=states,
                actions=actions,
                next_states=next_states,
                rewards=rewards,
                dones=dones,
                probs = action_probs)
        
        


    def _learn_step(self):
        states,actions,rewards_future,next_states,dones,old_probs = self.sample()

        # normalize rewards
        rewards_mean = np.mean(rewards_future, axis=1)
        rewards_std = np.std(rewards_future, axis=1) + 1.0e-10
        rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]

