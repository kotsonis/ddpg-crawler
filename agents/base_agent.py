"""base for RL Agent classes.

usage : foo = Agent(**kwargs)
        foo.train(num_iterations)
        foo.play(num_episodes)
"""
import os
import random
import numpy as np
from tqdm import tqdm
from tqdm import trange
from absl import logging
from absl import flags
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from collections import deque
import copy

from networks import models
from buffers import PrioritizedReplayBuffer, nStepPER

config = flags.FLAGS
flags.DEFINE_float(
    name='actor_lr',
    default=3e-4,
    help='lr for actor optimizer')
flags.DEFINE_float(
    name='critic_lr',
    default=3e-4,
    help='lr for critic optimizer')
flags.DEFINE_float(
    name='max_frames_per_episode',
    default=1000,
    help='maximum number of frames to process per episode')
flags.DEFINE_string(
    name='load_model',
    default='./model/model_saved.pt',
    help='saved agent model to load')
flags.DEFINE_integer(
    name='training_iterations',
    default=1e5,
    help='number of agent/env interactions to perform')
flags.DEFINE_integer(
    name='learn_every',
    default=4,
    help='number of environment interactions for every training step')

def tile(a, dim, n_tile):
    """creates torch.tensor by repeating a dimension n_tile times."""
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)
class Agent():
    """Base Agent for RL."""
    def __init__(
                self,
                env,
                *pargs,
                **kwargs):
        self.device = kwargs.setdefault('device','cpu')
        self.name = kwargs.setdefault('name','BaseRLAgent')
        self.learn_every = kwargs.setdefault('learn_every',config.learn_every)

        self.logger = logging.getLogger("Agent").setLevel(logging.DEBUG)
        logger = self.logger
        logger.info('Create an Agent type %s', __name__)
        
        #process Unity environment details
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[env.brain_names[0]]
        self.da = kwargs.setdefault("action_size",self.brain.vector_action_space_size)
        self.ds = kwargs.setdefault("state_size",self.brain.vector_observation_space_size)
        # create replay buffer
        replay_buffer_class = kwargs.setdefault('replay_buffer_class',PrioritizedReplayBuffer)
        self.memory = replay_buffer_class(**kwargs)
        
        #create actor & critic networks
        actor_dnn_class = kwargs.setdefault('actor_dnn_class',models.Actor)
        self.actor = actor_dnn_class(**kwargs).to(self.device)
        self.target_actor = actor_dnn_class(**kwargs).to(self.device)
        critic_dnn_class = kwargs.setdefault('critic_dnn_class',models.Critic)
        self.critic = critic_dnn_class(**kwargs).to(self.device)
        self.target_critic = critic_dnn_class(**kwargs).to(self.device)

        # copy network parameters from online to target
        for target_param, param in zip(
                self.target_actor.parameters(),
                self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        for target_param, param in zip(
                self.target_critic.parameters(),
                self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        
        # setup optimizers
        actor_optimizer = kwargs.setdefault('actor_optimizer',torch.optim.Adam)
        actor_optimizer_lr = config.actor_lr
        self.actor_optimizer = actor_optimizer(
                                              self.actor.parameters(),
                                              lr=actor_optimizer_lr)
        
        critic_optimizer = kwargs.setdefault('critic_optimizer',torch.optim.Adam)
        critic_optimizer_lr = config.critic_lr
        self.critic_optimizer = critic_optimizer(
                                                self.critic.parameters(),
                                                lr=critic_optimizer_lr)
        
        # common training / playing parameters
        self.max_frames_per_episode = config.max_frames_per_episode

        self.iteration = 0

    def act(self,states, add_noise=True):
        """ act according to target policy based on state """

        # move states into torch tensor on device
        state = torch.FloatTensor(states).to(device)
        # turn off training mode
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy()
            # if we are being stochastic, add noise weighted by exploration
            if add_noise:
                action += self.eps*self.noise.sample()
        self.actor_target.train()

        return np.clip(action, -1, 1)  # TODO: clip according to Unity environment feedback
    
    def step(self, state, action, reward, next_state, done):
            """Processes one step of Agent/environment interaction and invokes a learning step accordingly."""

        # Save experience / reward
        self.memory.add(state,action,reward,next_state,done)
        self.t_step += 1
        
        if (self.t_step < 2000) : return self.train_step
        # Learn every UPDATE_EVERY time steps.
        if (self.t_step + 1) % self.learn_every == 0:
            # check if there are enough samples in memory for training
            if self.memory.enough_samples():
                self.learn()
        return self.train_step

    def play(self,
            episodes=10,
            max_frames_per_episode=1000,
            **kwargs):
        """plays an environment with model for requested episodes."""
        logger = self.logger
        self.sample_episode_maxlen = frames
        self.replaybuffer.clear()
        buff = []
        logger.info('Starting a Play session')
        logger.info('Will run for %d episodes', episodes)
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        num_agents = len(env_info.agents)
        states = env_info.vector_observations
        scores = [] # list containing scores from each episode
        agent_scores = np.zeros(num_agents)
        
        for episode in range(0, episodes):
            # initialize the score (for each agent)
            agent_scores = np.zeros(num_agents)
            frames = 0
            for steps in range(self.max_frames_per_episode):
                    actions = self.target_actor.action(
                        torch.from_numpy(states).type(
                            torch.float32).to(self.device)
                    ).cpu().numpy()
                    env_info = self.env.step(actions)[self.brain_name]
                    next_states = env_info.vector_observations
                    rewards = env_info.rewards
                    # fix NaN rewards of crawler environment, by penalizing a NaN reward
                    rewards = np.nan_to_num(rewards, nan=-5.0)
                    dones = env_info.local_done                       # see if episode finished
                    frames += 1
                    agent_scores += rewards
                    if frames % 20 == 0:
                        print('\rEpisode {}\t Frame: {:4}/1000 \t Score: {:.2f}'
                            .format(
                                episode,
                                frames,
                                np.mean(agent_scores)
                            ), end="")
                    if np.any(dones):
                        break
                    else:
                        states = next_states
                scores.append(np.mean(agent_scores))
                print('\rEpisode: {}\tscore:{}\t running mean score: {:.2f}'
                    .format(
                        episode,
                        np.mean(agent_scores),
                        np.mean(scores)))
    def train(self, **kwargs):
        self.training_iterations = kwargs.setdefault('training_iterations', config.training_iterations)
        env = self.env
        scores = []                                 # list containing scores from each episode
        scores_window = deque(maxlen=100)           # last 100 scores
        agent_scores = np.zeros(num_agents)         # initialize the score (for each agent)
        
        i_episode = 0
        prev_iteration = 0
        save_counter = 0
        t_step = 0
        step = 0
        total_progress = tqdm(
                             total=self.training_iterations,
                             desc='step {} mean reward {:3.2f}'.format(self.iteration, np.mean(scores_window)))
        
        
        while self.iteration < self.training_iterations:
            for it in trange(15, leave=False, desc='Episodes {:3d}-{:3d}'.format(i_episode, i_episode+15)):
                i_episode += 1
                env_info = env.reset(train_mode=True)[brain_name] # reset the environment
                states = env_info.vector_observations             # get the current state of each agent
                agent_scores = np.zeros(num_agents)
                agent.noise.reset()
                while True: # each frame
                    actions = agent.act(states, add_noise=True)
                    step += 1
                    env_info = env.step(actions)[brain_name]          # send all actions to tne environment
                    next_states = env_info.vector_observations        # get next state (for each agent)
                    rewards = env_info.rewards                        # get reward (for each agent)
                    if (np.any(np.isnan(rewards))): 
                        print('got a NaN reward. Need to fix it.')
                    rewards = np.nan_to_num(rewards,nan=-5.0)
                    dones = env_info.local_done                       # see if episode finished
                    #if frames < 999: rewards += np.array(dones)*-5.0
                    self.step(states, actions, rewards, next_states, dones)
                
                    # update progress bar
                    if (self.iteration % 10):
                        total_progress.update(self.iteration-prev_iteration)
                        prev_iteration = self.iteration
                    t_step_prev = t_step
                    # saving and logging
                    if self.iteration % model_save_period == 0:
                        self.save_model(os.path.join(model_save_dir, 'model_{:4d}.pt'.format(save_counter)))
                        save_counter += 1
                    self.write_tensorboard()
                    agent_scores += rewards                           # update the score (for each agent)
                    states = next_states                              # roll over states to next time step
                    if np.any(dones):                                 # exit loop if episode finished
                        break
                
                # save latest model
                self.save_model(os.path.join(model_save_dir, 'model_latest.pt'))
                
                scores.append(np.mean(agent_scores))              # store episodes mean reward over agents
                scores_window.append(np.mean(agent_scores))       # save most recent score
            
            if i_episode % 100 == 0:
                print('\rEpisodes: {}\t 100 episode mean score: {:.2f}\t training to go: {:.0f}, eps: {:.2f}'.format(
                    i_episode, np.mean(scores_window), total_train_steps - t_step, agent.eps ))
                    agent.save_models(hyper_params.model_dir)
                if np.mean(scores_window)>=solution:
                    if ( not solution_found):
                        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                        solution_num_episodes = i_episode-100
                        solution_found = True
                        break
            
    def write_tensorboard(self):
        """Writes training data to tensorboard summary writer. Should be overloaded by sub-classes."""
        it = self.iteration
        writer.add_scalar('return', self.mean_return, it)
        writer.add_scalar('reward', self.mean_reward, it)
        writer.add_scalar('loss_q', self.mean_loss_q, it)
        writer.add_scalar('loss_p', self.mean_loss_p, it)
        writer.add_scalar('loss_l', self.mean_loss_l, it)
        writer.add_scalar('mean_q', self.mean_est_q, it)
        writer.flush()

    def load_model(self, **kwargs):
            """
            loads a model from a given path
            :param path: (str) file path (.pt file)
            """
            load_path = kwargs.setdefault(
                                        'path',
                                        config.load_model)
            checkpoint = torch.load(load_path)
            self.iteration = checkpoint['iteration']
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
            self.critic.train()
            self.target_critic.train()
            self.actor.train()
            self.target_actor.train()
            self.logger.info('Loaded model: {}'.format(load_path))
    def save_model(self, path=None):
        """Saves the model.
        :param path: (str) file path (.pt file)
        """
        data = {
            'iteration': self.iteration,
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict()
        }
        torch.save(data, path)
    def _soft_update(self, local_model, target_model):
        """Updates target model parameters with local.

        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    
