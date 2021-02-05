""" RL Agent based on Proximal Policy Optimization  """
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.OUNoise import OUNoise
import copy
import networks
from agents.base import Agent
from utils import replay
from absl import logging
from absl import flags
from collections import deque
from tqdm import tqdm
from tqdm import trange

config = flags.FLAGS
flags.DEFINE_integer(name='PPO_iterations',default=500,
    help='number of iteration steps to perform per sample in PPO')
class PPOAgent():
    """PPO Agent """
    def __init__(
                self,
                env,
                *pargs,
                **kwargs):
        self.device = kwargs.setdefault('device','cpu')
        self.name = kwargs.setdefault('name','PPOAgent')
        self.gamma = kwargs.setdefault('gamma', config.gamma)
        logging.info('Create an Agent type %s', self.name)
        self.eps_start = kwargs.get('eps_start', config.eps_start)
        self.eps_minimum = kwargs.get('eps_minimum', config.eps_minimum)
        self.eps_decay = kwargs.get('eps_decay', config.eps_decay)
        self.eps = self.eps_start

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
        low_bound = np.ones((self.da))*-0.99999
        upper_bound = np.ones((self.da))*0.99999
        self.action_bounds = (low_bound,upper_bound)
        kwargs['action_bounds'] = self.action_bounds
        hidden_dims=(512,256) # (512,256,256)
        kwargs['hidden_dims'] = hidden_dims
        # create replay buffer
        #self.memory = replay.PPOBuffer(**kwargs)
        activation_fc= F.relu # torch.tanh # lambda x: F.leaky_relu(x, negative_slope=0.5) # F.gelu # F.relu
        kwargs['activation_fc'] = activation_fc
        #create policy
        self.policy = networks.distributional.PolicyPPO(**kwargs)
        self.value = networks.distributional.ValuePPO(**kwargs)
        lrate_schedule = lambda it: max(0.995 ** it, 0.005)
        
        # setup optimizers
        actor_optimizer = kwargs.setdefault('actor_optimizer',torch.optim.Adam)
        actor_optimizer_lr = kwargs.get('actor_lr',config.actor_lr)
        amsgrad = False
        self.policy_optimizer = actor_optimizer(self.policy.parameters(),
                                                lr=actor_optimizer_lr,
                                                amsgrad=amsgrad)
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda=lrate_schedule)
        critic_optimizer = kwargs.setdefault('critic_optimizer',torch.optim.Adam)
        critic_optimizer_lr = kwargs.get('critic_lr',config.critic_lr)

        self.value_optimizer = critic_optimizer(self.value.parameters(),
                                                lr=critic_optimizer_lr,
                                                amsgrad=amsgrad)
        self.value_scheduler = torch.optim.lr_scheduler.LambdaLR(self.value_optimizer, lr_lambda=lrate_schedule)
        # common training / playing parameters
        
        self.max_frames_per_episode = config.max_frames_per_episode
        self.model_save_period = 500
        self.iteration = 0
        self.saved_iteration = 0
        self.log_dir = kwargs.setdefault('log_dir', config.log_dir)
        self.model_save_dir = os.path.join(self.log_dir, 'model')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        self._tb_init()
        
        self.tot_epochs = 0
        self.episodes = 0
        self.episodes_rewards = deque(maxlen=100)

        self.policy_optimization_epochs = 20 #80  # 30
        self.value_optimization_epochs = 20 #80 # 30
        self.policy_sampling_ratio = self.value_sampling_ratio = 0.3
        # self.policy_sampling_ratio = self.value_sampling_ratio = 0.5
        self.ppo_early_stop = False
        self.policy_stopping_kl = 15 # 2.5 #0.5 # 0.02
        if (self.ppo_early_stop):
            self.policy_stopping_kl_fn = lambda x: x> self.policy_stopping_kl
        else:
            self.policy_stopping_kl_fn = lambda x: False
        self.value_stopping_mse = 10 # 0.5
        self.policy_clip_range =  0.2
        self.value_clip_range = float('inf')
        self.gae_tau = 0.97
        self.beta = 0.001 #0.2 # 0.001 #0.01
        self.policy_gradient_clip = float('inf') #1.0
        self.value_gradient_clip = float('inf')
        self.start_random_steps = 0
        self.num_start_steps = 1000
    def _next_eps(self):
        """updates exploration factor"""
        self.eps = max(self.eps_minimum, self.eps*self.eps_decay)
    
    def collect_trajectories(self, num_trajectories):
        agent_scores = np.zeros(self.num_agents)
        horizon = deque()

        env = self.env
        env_info = env.reset(train_mode=True)[self.brain_name]
        
        for _ in range(num_trajectories):
            states = env_info.vector_observations
            self.start_random_steps += 1
            actions, log_probs = self.policy.np_action(states, eps= 0.0) # self.eps)

            values = self.value(states).detach()
            # send all actions to tne environment and collect observation
            env_info = env.step(actions)[self.brain_name]     
            rewards = np.array(env_info.rewards)                      
            dones = np.array(env_info.local_done)
            max_reached = env_info.max_reached 
            # update the score (for each agent)
            agent_scores += rewards                           
            # check if any agents finished an episode
            for i,d in enumerate(dones):
                if d:
                    self.episodes_rewards.append(agent_scores[i])
                    agent_scores[i] = 0.0
                    self.episodes += 1
            horizon.append((states,actions,rewards,dones,log_probs,values))
            # if np.any(max_reached) is True: env_info = env.reset(train_mode=True)[self.brain_name]
        
        # decrease exploration ratio
        self._next_eps()

        # process horizon backwards for generalized advantage

        next_values = self.value(states).detach().unsqueeze(-1)
        # push next value at end of horizon
        # trajectory = [None] * (len(trajectory_raw)-1)
        # process raw trajectories
        # calculate advantages and returns
        gae = torch.zeros(self.num_agents, 1).to(self.device)
        # returns = next_values.detach()
        next_v = next_values.detach()
        trajectory = [None]*(num_trajectories)
        i = 0
        while (len(horizon) > 0):
            states,actions,rewards,dones,log_probs,values = horizon.pop()
            # turn np arrays into tensors
            states, actions, rewards, dones, log_probs = map(
                lambda x: torch.tensor(x).float().to(self.device),
                (states, actions, rewards, dones, log_probs)
            )
            ongoing = (1-dones).unsqueeze(-1)
            rewards = rewards.unsqueeze(-1)
            values = values.unsqueeze(-1)
            delta = rewards + self.gamma*next_v*ongoing - values
            gae = delta + self.gae_tau*self.gamma*ongoing*gae
            returns = gae + values 
            
            # calculate advantage
            #td_errors = rewards + self.gamma * ongoing * next_values - values
            #gae = gae * self.gae_tau * self.gamma * ongoing + td_errors

            # store in trajectory list
            trajectory[i] = (states, actions, log_probs, returns, gae, values)
            i += 1
            next_v = values
        
        # pack trajectories
        states, actions, old_log_probs, returns, gae, values = map(
            lambda x: torch.cat(x, dim=0), zip(*trajectory)
            )
        adv = returns - values
        # normalize advantages
        adv = (adv - adv.mean())  / (adv.std() + 1.0e-6)
        # return data for subsequent sampling
        return states, actions, old_log_probs, returns, adv

    def train(self,**kwargs):
        self.training_iterations = kwargs.pop('training_iterations', config.training_iterations)
        max_episodes = kwargs.setdefault('PPO_iterations',config.PPO_iterations)
        self.training_iterations += self.saved_iteration  # in case we are continuing training

        env = self.env
        scores = []                                 # list containing scores from each episode
        scores_window = deque(maxlen=100)           # last 100 scores
        fpe_window = deque(maxlen=100)
        num_agents = self.num_agents
        agent_scores = np.zeros(num_agents)         # initialize the score (for each agent)
        
        
        
        fpe = 0.0 # frames per episode
        prev_iteration = 0
        save_counter = 0
        self.t_step = 0
        step = 0
        total_progress = tqdm(
                             total=self.training_iterations-self.iteration,
                             desc='step {}, epochs: {}, episodes: {}, mean score:{:3.2f},'.format(
                                                    self.iteration, self.tot_epochs, self.episodes,self.mean_return))
        
        solution_found = False
        solution = 2000
        prev_iteration = self.iteration
        self.min_ppo_epochs = 5
        while self.iteration < self.training_iterations:
            all_states, all_actions, all_old_log_probs, all_returns, all_gae = self.collect_trajectories(max_episodes)
            all_values = self.value(all_states).detach()
            n_samples = len(all_returns)
            # normalize returns
            epoch = 0
            
        
            # optimize policy
            for policy_epochs in range(self.policy_optimization_epochs):
                self.tot_epochs += 1
                # sample a batch
                with torch.no_grad():
                    batch_size = int(self.policy_sampling_ratio*n_samples)
                    idx = np.random.choice(n_samples, batch_size, replace=False)
                    states = all_states[idx]
                    actions = all_actions[idx]
                    old_log_probs = all_old_log_probs[idx]
                    returns = all_returns[idx]
                    gae = all_gae[idx]
                    # returns = (returns - returns.mean())  / (returns.std() + 1.0e-6)
                    # gae = (gae - gae.mean())  / (gae.std() + 1.0e-6)
                    
                log_probs, entropy = self.policy.get_probs(states,actions)

                # probability ratio
                ratio = (log_probs - old_log_probs).exp()
                ratio_clamped = ratio.clamp(1-self.policy_clip_range, 1+self.policy_clip_range)

                policy_objective = gae*ratio
                policy_objective_clamped = gae*ratio_clamped

                policy_loss = -torch.min(policy_objective, policy_objective_clamped).mean()
                entropy_loss = -entropy.mean()*self.beta

                self.policy_optimizer.zero_grad()
                (policy_loss+entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.policy_gradient_clip)
                self.policy_optimizer.step()
                epoch += 1
                with torch.no_grad():
                    # check if we can break early
                    #if epoch > self.min_ppo_epochs:
                    log_probs_all, _ = self.policy.get_probs(all_states, all_actions)
                    kl = (all_old_log_probs - log_probs_all).mean()
                    if self.policy_stopping_kl_fn(kl.item()):
                        break
            # store info for tensorboard after policy optimization epochs
            with torch.no_grad():
                self.surrogate_loss = policy_objective.mean()
                self.mean_loss_policy = policy_loss
                self.mean_loss_entropy = entropy_loss
                self.mean_entropy = entropy.mean()
                self.kl_distance = kl.item()
            
            
            # optimize value
            for value_epochs in range(self.value_optimization_epochs):
                # sample a batch
                batch_size = int(self.value_sampling_ratio*n_samples)
                idx = np.random.choice(n_samples, batch_size, replace=False)
                states = all_states[idx]
                returns = all_returns[idx].squeeze(-1)
                values = all_values[idx]
               
                values_pred = self.value(states)

                #values_pred_clipped = values + (values_pred - values).clamp(-self.value_clip_range, 
                #                                                             self.value_clip_range)
                # v_loss = (returns - values_pred).pow(2)
                # v_loss_clipped = (returns - values_pred_clipped).pow(2)
                value_loss = F.smooth_l1_loss(values_pred,returns)
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), self.value_gradient_clip)
                self.value_optimizer.step()
                
                with torch.no_grad():
                    # check if we can break early
                    val_pred_all = self.value(all_states)
                    mse = (all_values - val_pred_all).pow(2).mul(0.5).mean()
                    if mse.item() > self.value_stopping_mse:
                        break
            # store info for tensorboard after policy optimization epochs
            with torch.no_grad():
                self.mean_loss_value = value_loss
                self.value_mse = mse.item()
                self.mean_value = val_pred_all.mean()
                self.value_epochs = value_epochs
                self.policy_epochs = policy_epochs
                
            # update progress bar
            if (self.iteration+1) % 1 == 0:
                self.mean_return = np.mean(self.episodes_rewards)
                total_progress.update(self.iteration-prev_iteration)
                total_progress.set_description('step {:4d}, epochs: {:4d}, episodes: {:4d}, mean score:{:3.2f},'.format(
                                                    self.iteration, self.tot_epochs, self.episodes, self.mean_return ))
                prev_iteration = self.iteration
            self._next_iter()
            self.value_scheduler.step()
            self.policy_scheduler.step()   
                
        

    def _learn_step(self):
        states,actions,rewards,next_states,dones,old_probs = self.sample()
        states = states.view(-1,self.ds).detach()
        self.B = states.shape[0]
        actions = actions.view(-1,self.da).detach()
        rewards = rewards.view(-1,1).detach()
        next_states = next_states.view(-1,self.ds).detach()
        dones = dones.view(-1,1)


        # normalize rewards
        rewards_mean = np.mean(rewards_future, axis=1)
        rewards_std = np.std(rewards_future, axis=1) + 1.0e-10
        rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]

    def save_model(self, path=None):
        """Saves the model."""
        data = {
            'iteration': self.iteration,
            'policy_state_dict': self.policy.state_dict(),
            'policy_optim_state_dict': self.policy_optimizer.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'value_optim_state_dict': self.value_optimizer.state_dict()
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
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optim_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optim_state_dict'])
            self.policy.train()
            self.value.train()
            logging.info('Loaded model: {}'.format(load_path))
            logging.info('iteration: {}'.format(self.iteration))

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
    def _tb_init(self):
        ts = datetime.datetime.now().replace(microsecond=0).strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'tb',ts))
        self.tensorboard_update_period = 1
        # tb tracking fields
        self.mean_loss_value = 0.0
        self.value_mse = 0.0
        self.surrogate_loss = 0.0
        self.mean_loss_policy = 0.0
        self.mean_loss_value = 0.0
        self.mean_loss_entropy = 0.0
        self.mean_value = 0.0
        self.kl_distance = 0.0
        self.mean_return = 0.0
        self.mean_entropy = 0.0
        self.value_epochs = 0
        self.policy_epochs = 0
        return

    def _tb_write(self):
        """Writes training data to tensorboard summary writer. Should be overloaded by sub-classes."""
        it = self.iteration
        # tb tracking fields
        self.writer.add_scalar('loss/clipped_surrogate', self.surrogate_loss, it)
        self.writer.add_scalar('loss/policy', self.mean_loss_policy, it)
        self.writer.add_scalar('loss/value', self.mean_loss_value, it)
        self.writer.add_scalar('loss/entropy', self.mean_loss_entropy, it)
        self.writer.add_scalar('value mse', self.value_mse,it)
        self.writer.add_scalar('mean/ return', self.mean_return, it)
        self.writer.add_scalar('mean/entropy', self.mean_entropy, it)
        self.writer.add_scalar('policies kl distance', self.kl_distance, it)
        self.writer.add_scalar('mean/trajectories value', self.mean_value, it)
        self.writer.add_scalar('epochs/value', self.value_epochs, it)
        self.writer.add_scalar('epochs/policy', self.policy_epochs, it)
        self.writer.flush()