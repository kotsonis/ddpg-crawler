"""base for RL Agent classes.

usage : foo = Agent(**kwargs)
        foo.train(num_iterations)
        foo.play(num_episodes)
        action(s) = foo.act(state(s))
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
from utils.OUNoise import OUNoise
import copy
import datetime
from networks import models
from utils.priority_buffers import PrioritizedReplayBuffer

# global config class to share command line/default parameters across modules
config = flags.FLAGS
flags.DEFINE_float(
    name='eps_start',
    default=1.0,
    help='starting exploration rate (0,1]')
flags.DEFINE_float(
    name='eps_minimum',
    default=0.001,
    help='minimum exploration rate')
flags.DEFINE_float(
    name='eps_decay',
    default=0.995,
    help='eps decay rate. eps=eps*eps_decay')
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
    default=100000,
    help='number of agent/env interactions to perform')
flags.DEFINE_integer(
    name='learn_every',
    default=4,
    help='number of environment interactions for every training step')
flags.DEFINE_float(
    name='gamma',
    default=0.99,
    help='discount factor for future rewards (0,1]')
flags.DEFINE_float(
    name='soft_update_tau',
    default = 0.001,
    help='soft update factor for copying online actor/critic into target')
def tile(a, dim, n_tile, device):
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
        self.eps_start = kwargs.get('eps_start', config.eps_start)
        self.eps_minimum = kwargs.get('eps_minimum', config.eps_minimum)
        self.eps_decay = kwargs.get('eps_decay', config.eps_decay)
        self.eps = self.eps_start
        self.tau = kwargs.get('soft_update_tau', config.soft_update_tau)
        self.learn_every = kwargs.setdefault('learn_every',config.learn_every)
        self.gamma = kwargs.setdefault('gamma', config.gamma)
        logging.info('Create an Agent type %s', __name__)
        self.save_counter = 0
        #process Unity environment details
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[env.brain_names[0]]
        self.da = kwargs.setdefault("action_size",self.brain.vector_action_space_size)
        self.ds = kwargs.setdefault("state_size",self.brain.vector_observation_space_size)
        # create replay buffer
        replay_buffer_class = kwargs.setdefault('replay_buffer_class',PrioritizedReplayBuffer)
        self.memory = replay_buffer_class(**kwargs)
        self.noise = OUNoise(self.da)
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
        self.model_save_period = 1000
        self.iteration = 0
        self.log_dir = kwargs.setdefault('log_dir', config.log_dir)
        self.model_save_dir = os.path.join(self.log_dir, 'model')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        ts = datetime.datetime.now().replace(microsecond=0).strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'tb',ts))
        self.tensorboard_update_period = 50
    def save_model(self, path=None):
        """Saves the model."""
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
    def load_model(self, **kwargs):
            """loads a model from a given path."""
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
        self.memory.compute_beta_decay(self.training_iterations)
        env = self.env
        scores = []                                 # list containing scores from each episode
        scores_window = deque(maxlen=100)           # last 100 scores
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        num_agents = len(env_info.agents)
        states = env_info.vector_observations
        agent_scores = np.zeros(num_agents)         # initialize the score (for each agent)
        
        self.mean_reward = 0
        self.mean_return = 0
        self.mean_loss_q = 0
        self.mean_loss_p = 0
        self.mean_loss_l = 0
        self.mean_est_q = 0
        self.min_td = 0
        self.max_td = 0

        i_episode = 0
        prev_iteration = 0
        save_counter = 0
        self.t_step = 0
        step = 0
        total_progress = tqdm(
                             total=self.training_iterations,
                             desc='step {} mean reward {:3.2f}'.format(self.iteration, self.mean_return))
        
        solution_found = False
        solution = 1200

        while self.iteration < self.training_iterations:
            #for it in trange(15, leave=False, desc='Episodes {:3d}-{:3d}'.format(i_episode, i_episode+15)):
            i_episode += 1
            #env_info = env.reset(train_mode=True)[self.brain_name] # reset the environment
            states = env_info.vector_observations             # get the current state of each agent
            agent_scores = np.zeros(num_agents)
                
            self.noise.reset()
            while True: # each frame
                actions = self.act(states, add_noise=True)
                step += 1
                env_info = env.step(actions)[self.brain_name]          # send all actions to tne environment
                next_states = env_info.vector_observations        # get next state (for each agent)
                rewards = env_info.rewards                        # get reward (for each agent)
                if (np.any(np.isnan(rewards))): 
                    print('got a NaN reward. Need to fix it.')
                rewards = np.nan_to_num(rewards,nan=-5.0)
                agent_scores += rewards                           # update the score (for each agent)
                self.mean_rewards = np.mean(agent_scores)
                dones = env_info.local_done                       # see if episode finished
                #if frames < 999: rewards += np.array(dones)*-5.0
                self.step(states, actions, rewards, next_states, dones)
                
                # update progress bar
                if (self.iteration+1) % 10 == 0:
                    total_progress.update(self.iteration-prev_iteration)
                    total_progress.set_description('step {} mean reward {:3.2f}'.format(self.iteration, self.mean_reward))
                    prev_iteration = self.iteration
                    
                states = next_states                              # roll over states to next time step
                if np.any(dones):                                 # exit loop if episode finished
                    break
                
            scores.append(np.mean(agent_scores))              # store episodes mean reward over agents
            self.mean_return = np.mean(agent_scores)
            scores_window.append(np.mean(agent_scores))       # save most recent score
            
            if i_episode % 100 == 0:
                print('\rEpisodes: {}\t 100 episode mean score: {:.2f}'.format(
                    i_episode, np.mean(scores_window)))
         
            if np.mean(scores_window)>=solution:
                if ( not solution_found):
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                    solution_num_episodes = i_episode-100
                    solution_found = True
                    break         

    def act(self,states, add_noise=True):
        """ act according to target policy based on state """
        # move states into torch tensor on device
        state = torch.FloatTensor(states).to(self.device)
        # turn off training mode
        self.target_actor.eval()
        with torch.no_grad():
            action = self.target_actor(state).cpu().data.numpy()
            # if we are being stochastic, add noise weighted by exploration
            if add_noise:
                action += self.eps*self.noise.sample().data.numpy()
        self.target_actor.train()
        return np.clip(action, -1, 1)  # TODO: clip according to Unity environment feedback

    def step(self, state, action, reward, next_state, done):
        """Processes one step of Agent/environment interaction and invokes a learning step accordingly."""
        # Save experience / reward
        self.memory.add(state=state,action=action,reward=reward,next_state=next_state,done=done)
        self.t_step += 1
        if (self.t_step < 2000) : return self.iteration
        # Learn every UPDATE_EVERY time steps.
        if (self.t_step + 1) % self.learn_every == 0:
            # check if there are enough samples in memory for training
            if self.memory.enough_samples():
                self._learn_step()
                self._next_iter()
        return self.iteration
    def _learn_step(self):
        """performs one learning step"""
        # ---------------- sample a batch of experiences ---------------------- #
        states,actions,rewards,next_states,dones, weights, indices = self.memory.sample()
        # decay beta for next sampling (Prioritized Experience Replay related)
        self.memory.decay_beta()
        
        # unravel S,A,R,S,done in case we have stored a multi-agent experience
        B = states.shape[0]
        states = states.view(-1,self.ds)
        actions = actions.view(-1,self.da)
        rewards = rewards.view(-1,1)
        next_states = next_states.view(-1,self.ds)
        dones = dones.view(-1,1)
        # -------------------- get target Q value ----------------------------- #
        future_action = self.target_actor(next_states)
        q_future_state = self.target_critic(next_states, future_action)
        q_future_state *= (1-dones) 
        q_target = rewards + self.gamma*q_future_state
        # -------------------- get current Q value ---------------------------- #
        q_online = self.critic(states,actions)
        # store for tensorboard and priority calculation
        with torch.no_grad():
            current_q = q_online.detach()
            self.mean_est_q = current_q.mean()
        # --------------------- compute critic loss --------------------------- #
        error_loss = q_target - q_online
        loss = F.smooth_l1_loss(q_target, q_online,reduction='none')
        loss = loss.view(B,-1)
        loss = loss.mean(dim=-1)
        critic_loss = (weights*loss.view(B,-1))
        critic_loss_batch = critic_loss.mean()
        # store loss for tensorboard
        with torch.no_grad():
            self.mean_loss_q = critic_loss_batch
        # -------------------- optimize the critic --------------------------- #
        self.critic_optimizer.zero_grad()
        critic_loss_batch.backward()
        self.critic_optimizer.step()
        # -------------------- compute actor loss ---------------------------- #
        # we want maximum reward (Q) so loss is negative of current Q
        actor_loss = - self.critic(states, self.actor(states)).mean(dim=1).mean()
        # store loss for tensorboard
        with torch.no_grad():
            self.mean_loss_p = actor_loss
        # ------------------- optimize the actor ----------------------------- #
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ------------------- Prioritized Replay Buffer ---------------------- #
        # ------------ update priorities in the replay buffer ---------------- #
        # calculate TD_error/Q. Avoid exploding priority through division by <1
        with torch.no_grad():
            current_q = torch.clamp_min(current_q,1.0)
            td_error = error_loss.abs()
            td_error = td_error/(current_q.abs()+1e-6)
            td_error = td_error.view(B,-1)
            td_error = td_error.mean(dim=1)
            self.min_td = td_error.min()
            self.max_td = td_error.max()
            new_p = td_error.cpu().data.numpy().clip(0,2.0)
            # ------------------ update PER priorities ----------------------- #
            self.memory.update_priorities(indices, new_p.tolist())
        # ------------------ soft update target networks --------------------- #
        self._soft_update(self.critic, self.target_critic)
        self._soft_update(self.actor, self.target_actor)
        # ------------------ hard update target networks --------------------- #
        if (((self.iteration +1) % 1000) == 0): 
            self._hard_update(self.critic, self.target_critic)
            self._hard_update(self.actor, self.target_actor)  
        # ----------------- update noise and exploration --------------------- #
        self._next_eps()
        return

    def _soft_update(self, local_model, target_model):
        """Updates target model parameters with local.

        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    def _hard_update(self,local_model,target_model):
        """copies local model parameters into target.

        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
    def _next_eps(self):
        """updates exploration factor"""
        self.eps = max(self.eps_minimum, self.eps*self.eps_decay)    
    def _next_iter(self):
        """increases training iterator and performs logging/model saving"""
        self.iteration += 1
        # save latest model
        self.save_model(os.path.join(self.model_save_dir, 'model_latest.pt'))
        if (self.iteration+1) % self.model_save_period ==0:
            self.save_model(os.path.join(self.model_save_dir, 'model_{:4d}.pt'.format(self.save_counter)))
            self.save_counter += 1
        if (self.iteration+1) % self.tensorboard_update_period ==0:
            self._tb_write()
        return self.iteration
    def _tb_write(self):
        """Writes training data to tensorboard summary writer. Should be overloaded by sub-classes."""
        it = self.iteration
        self.writer.add_scalar('return', self.mean_return, it)
        self.writer.add_scalar('eps', self.eps, it)
        self.writer.add_scalar('loss_q', self.mean_loss_q, it)
        self.writer.add_scalar('loss_p', self.mean_loss_p, it)
        self.writer.add_scalar('min_td', self.min_td, it)
        self.writer.add_scalar('max_td', self.max_td, it)
        self.writer.add_scalar('mean_q', self.mean_est_q, it)
        self.writer.flush()