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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from utils.OUNoise import OUNoise
import copy
import datetime
import networks
from utils import replay
from utils import accumulator
from absl import logging
from absl import flags

config = flags.FLAGS

# global config class to share command line/default parameters across modules

flags.DEFINE_float(name='eps_start',default=1.0,
    help='starting exploration rate (0,1]')
flags.DEFINE_float(name='eps_minimum',default=0.001,
    help='minimum exploration rate')
flags.DEFINE_float(name='eps_decay',default=0.99,
    help='eps decay rate. eps=eps*eps_decay')
flags.DEFINE_float(name='actor_lr',default=2e-4,
    help='lr for actor optimizer')
flags.DEFINE_float(name='critic_lr',default=2e-4,
    help='lr for critic optimizer')
flags.DEFINE_float(name='max_frames_per_episode',default=1000,
    help='maximum number of frames to process per episode')
flags.DEFINE_string(name='load_model',default='./model/model_saved.pt',
    help='saved agent model to load')
flags.DEFINE_integer(name='training_iterations',default=100000,
    help='number of agent/env interactions to perform')
flags.DEFINE_integer(name='learn_every',default=4,
    help='number of environment interactions for every training step')
flags.DEFINE_float(name='gamma',default=0.99,
    help='discount factor for future rewards (0,1]')
flags.DEFINE_float(name='soft_update_tau',default = 0.001,
    help='soft update factor for copying online actor/critic into target')
flags.DEFINE_bool(name='episodic',default = True,
    help='train on episodes or max_frames_per_episode(True)')

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
        self.episodic = kwargs.get('episodic', config.episodic)
        self.hasPER = False # assume no priority replay
        self.sample_size = 0
        logging.info('Create an Agent type %s', __name__)
        self.save_counter = 0
        #process Unity environment details
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[env.brain_names[0]]
        self.da = kwargs.setdefault("action_size",self.brain.vector_action_space_size)
        self.ds = kwargs.setdefault("state_size",self.brain.vector_observation_space_size)

        self.env_info = self.env.reset(train_mode=False)[self.brain_name]
        self.num_agents = len(self.env_info.agents)
        kwargs['num_agents'] = self.num_agents
        # create replay buffer
        replay_buffer_class = kwargs.pop('replay_buffer_class',replay.Buffer)
        self.memory = replay_buffer_class(**kwargs)
        
        self.noise = OUNoise(self.da)
        #create actor & critic networks
        actor_dnn_class = kwargs.pop('actor_dnn_class',networks.greedy.Actor)
        self.actor = actor_dnn_class(**kwargs).to(self.device)
        self.target_actor = actor_dnn_class(**kwargs).to(self.device)
        critic_dnn_class = kwargs.pop('critic_dnn_class',networks.greedy.Critic)
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
        self.saved_iteration = 0
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
                                        'load_model',
                                        config.load_model)
            checkpoint = torch.load(load_path,map_location=torch.device(self.device))
            
            self.saved_iteration = checkpoint['iteration']
            self.iteration += self.saved_iteration
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
            logging.info('Loaded model: {}'.format(load_path))
            logging.info('iteration: {}'.format(self.iteration))
    def play(self,
            episodes=10,
            max_frames_per_episode=1000,
            **kwargs):
        """plays an environment with model for requested episodes."""
        buff = []
        logging.info('Starting a Play session')
        logging.info('Will run for %d episodes', episodes)
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        num_agents = len(env_info.agents)
        states = env_info.vector_observations
        scores = [] # list containing scores from each episode
        agent_scores = np.zeros(num_agents)
        
        for episode in range(0, episodes):
            # initialize the score (for each agent)
            agent_scores = np.zeros(num_agents)
            frames = 0
            for steps in range(max_frames_per_episode):
                actions = self.target_actor.action(
                    torch.from_numpy(states).type(
                        torch.float32).to(self.device), noise=False
                ).cpu().numpy()
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = np.array(env_info.rewards)
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
                if np.any(dones) == 1:
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
        self.training_iterations = kwargs.pop('training_iterations', config.training_iterations)
        
        self.training_iterations += self.saved_iteration  # in case we are continuing training
        try:
            self.memory.compute_beta_decay(self.training_iterations)
        except AttributeError:
            pass 

        env = self.env
        scores = []                                 # list containing scores from each episode
        scores_window = deque(maxlen=100)           # last 100 scores
        fpe_window = deque(maxlen=100)
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        num_agents = len(env_info.agents)
        states = env_info.vector_observations
        agent_scores = np.zeros(num_agents)         # initialize the score (for each agent)
        
        self.mean_reward=self.mean_return=self.mean_loss_q =self.mean_loss_p =self.mean_loss_l =self.mean_est_q =self.min_td = self.max_td = 0
        fpe = 0.0 # frames per episode
        i_episode = 0
        prev_iteration = 0
        save_counter = 0
        self.t_step = 0
        step = 0
        total_progress = tqdm(
                             total=self.training_iterations-self.iteration,
                             desc='step {},Episodes:{},mean score:{},frames/episode:{}'.format(
                                                    self.iteration, i_episode,self.mean_return, fpe))
        
        solution_found = False
        solution = 2000
        prev_iteration = self.iteration
        explore = True
        while self.iteration < self.training_iterations:
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations             # get the current state of each agent
            agent_scores = np.zeros(num_agents)
            frames = 0
            self.noise.reset()
            while frames <= (self.max_frames_per_episode+1): # each frame
                actions = self.act(states, add_noise=explore)
                step += 1
                frames += 1
                env_info = env.step(actions)[self.brain_name]          # send all actions to tne environment
                next_states = np.array(env_info.vector_observations)        # get next state (for each agent)
                rewards = np.array(env_info.rewards)                        # get reward (for each agent)
                dones = np.array(env_info.local_done)                       # see if episode finished
                agent_scores += rewards                           # update the score (for each agent)
                if (np.any(np.isnan(rewards)) or np.any(np.isnan(next_states)) or np.any(np.isnan(dones))):
                    break
                # normalize rewards as pct                
                #if frames < 999: rewards += np.array(dones)*-5.0
                self.step(state=states,action=actions,reward=rewards,next_state=next_states,done=dones)
                
                # update progress bar
                if (self.iteration+1) % 10 == 0:
                    total_progress.update(self.iteration-prev_iteration)
                    total_progress.set_description('step {},mean score:{:3.2f},frames/episode:{:9.2f}'.format(
                                                    self.iteration, self.mean_return, np.mean(fpe_window)))
                    prev_iteration = self.iteration
                    
                states = next_states                              # roll over states to next time step
            episodes = self.memory.clear_queue()

            fpe_window.append(frames/episodes)
            scores.append(np.mean(agent_scores))              # store episodes mean reward over agents
            self.mean_return = np.mean(agent_scores)
            scores_window.append(np.mean(agent_scores))       # save most recent score
            
         
            if np.mean(scores_window)>=solution:
                if ( not solution_found):
                    print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(self.iteration, np.mean(scores_window)))
                    solution_num_episodes = self.iteration
                    solution_found = True
                    break         

    def act(self,states, add_noise=True):
        """ act according to target policy based on state """
        # move states into torch tensor on device
        state = torch.FloatTensor(states, device=self.device)
        # turn off training mode
        #self.target_actor.eval()
        with torch.no_grad():
            action,_ = self.target_actor.action(state, self.eps)
            # if we are being stochastic, add noise weighted by exploration
        #self.target_actor.train()
        return np.clip(action.data.numpy(), -1, 1)  # TODO: clip according to Unity environment feedback

    def step(self, state, action, reward, next_state, done):
        """Processes one step of Agent/environment interaction and invokes a learning step accordingly."""
        # Save experience / reward
        self.memory.add(states=state,actions=action,rewards=reward,next_states=next_state,dones=done)
        self.t_step += 1
        if (self.t_step < 2000) : return self.iteration
        # Learn every UPDATE_EVERY time steps.
        if (self.t_step + 1) % self.learn_every == 0:
            # check if there are enough samples in memory for training
            if self.memory.ready:
                self._learn_step()
                self._next_iter()
        return self.iteration
    
    def sample(self):
        """samples from replay buffer.

        provides dummy data for weights, indexes if not available in underlying buffer"""
        self.memory.sample()
        states = self.memory.states
        actions = self.memory.actions
        rewards = self.memory.rewards
        next_states = self.memory.next_states
        dones = self.memory.dones
        self.sample_size = len(dones) # get batch size

        # get gammas from possible an n-step return buffer
        try:
            gammas = self.memory.gammas
        except AttributeError:
            gammas = torch.ones_like(dones, requires_grad=False)*self.gamma
        # get weights and indexes from possibly a prioritized buffer
        try:
            weights = self.memory.weights
            idxs = self.memory.idxes
            self.memory.decay_beta() # decay the beta factor in the PER since we just sampled
            self.hasPER = True
        except AttributeError:
            weights = torch.ones_like(dones, requires_grad=False)
            idxs = []
        try:
            probs = self.memory.probs
        except AttributeError:
            probs = torch.ones_like(dones)

        return states, actions, rewards, next_states, dones, gammas, probs, weights, idxs


    def _learn_step(self):
        """performs one learning step"""
        # ---------------- sample a batch of experiences ---------------------- #
        states,actions,rewards,next_states,dones, gammas, weights, indices = self.sample()
        
        # unravel S,A,R,S,done in case we have stored a multi-agent experience
        
        states = states.view(-1,self.ds)
        actions = actions.view(-1,self.da)
        rewards = rewards.view(-1,1)
        next_states = next_states.view(-1,self.ds)
        dones = dones.view(-1,1)
        # -------------------- get target Q value ----------------------------- #
        future_action = self.target_actor(next_states)
        q_future_state = self.target_critic(next_states, future_action)
        q_future_state *= (1-dones) 
        q_target = rewards + gammas*q_future_state
        # -------------------- get current Q value ---------------------------- #
        q_online = self.critic(states,actions)
        # store for tensorboard and priority calculation
        with torch.no_grad():
            current_q = q_online.detach()
            self.mean_est_q = current_q.mean()
        # --------------------- compute critic loss --------------------------- #
        error_loss = q_target - q_online
        loss = F.smooth_l1_loss(q_target, q_online,reduction='none')
        loss = loss.view(self.B,-1)
        loss = loss.mean(dim=-1)
        critic_loss = (weights*loss.view(self.B,-1))
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
        if (self.hasPER):
            with torch.no_grad():
                
                td_error =  error_loss.view(self.sample_size).mean(dim=1)
                td_error = td_error.abs()
                self.min_td = td_error.min()
                self.max_td = td_error.max()
                new_p = td_error.cpu().data.numpy()
                # ------------------ update PER priorities ----------------------- #
                self.memory.update_priorities(indices, new_p)

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
        
        if (self.iteration+1) % self.model_save_period ==0:
            self.save_model(os.path.join(self.model_save_dir, 'model_{:4d}.pt'.format(self.save_counter)))
            self.save_counter += 1
        if (self.iteration+1) % self.tensorboard_update_period ==0:
            # save latest model
            self.save_model(os.path.join(self.model_save_dir, 'model_latest.pt'))
            # write to tb log
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