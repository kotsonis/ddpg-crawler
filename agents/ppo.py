""" RL Agent based on Proximal Policy Optimization  """
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import copy
import networks
from absl import logging
from absl import flags
from collections import deque
from tqdm import tqdm
from tqdm import trange

config = flags.FLAGS

flags.DEFINE_float(name='actor_lr',default=2e-4,
    help='lr for actor optimizer')
flags.DEFINE_string(name='load_model',default='./model/model_saved.pt',
    help='saved agent model to load')
flags.DEFINE_integer(name='training_iterations',default=1000,
    help='number of agent/env interactions to perform')
flags.DEFINE_float(name='gamma',default=0.99,
    help='discount factor for future rewards (0,1]')
flags.DEFINE_integer(name='trajectories',default=2048,
    help='number of trajectories to sample per iteration')
flags.DEFINE_integer(name='policy_optimization_epochs', default=160,
    help='number of epochs to run (K in paper)')
flags.DEFINE_float(name='policy_stopping_kl', default=0.3,
    help='log KL divergence to early stop PPO improvements')
flags.DEFINE_float(name='policy_clip_range', default=0.2,
    help='clipping threshold for PPO policy optimization')
flags.DEFINE_float(name='gae_lambda', default=0.85,
    help='lambda coefficient for generalized advantage estimate')
flags.DEFINE_float(name='entropy_beta', default=0.002,
    help='coefficient to multiply beta loss in PPO step')
flags.DEFINE_float(name='vf_coeff', default=0.01,
    help='coefficient to multiply value loss in PPO step')
flags.DEFINE_integer(name='memory_batch_size',default=128,
    help='batch size of memory samples per epoch')
flags.DEFINE_bool(name='tb', default=True,
    help='enable tensorboard logging')
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

        # limit action bounds to less than 1 (so that arctanh does not return +- infinity)
        low_bound = np.ones((self.da))*-0.99999
        upper_bound = np.ones((self.da))*0.99999
        self.action_bounds = (low_bound,upper_bound)
        kwargs['action_bounds'] = self.action_bounds
        # set DNN hidden layers dimensions
        hidden_dims=(600,100)
        kwargs['hidden_dims'] = hidden_dims
        # select activation function
        activation_fc= torch.tanh
        kwargs['activation_fc'] = activation_fc
        kwargs['log_std_min'] = -22
        #create policy
        self.policy = networks.distributional.PPO(**kwargs)
        # setup optimizers
        actor_optimizer = kwargs.setdefault('actor_optimizer',torch.optim.Adam)
        actor_optimizer_lr = kwargs.get('actor_lr',config.actor_lr)
        amsgrad = False
        self.policy_optimizer = actor_optimizer(self.policy.parameters(),
                                                lr=actor_optimizer_lr,
                                                amsgrad=amsgrad)
        lrate_schedule = lambda it: max(0.995 ** it, 0.5) #0.005)
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda=lrate_schedule)
        
        # common training / playing parameters
        self.model_save_period = 500
        self.iteration = 0
        self.saved_iteration = 0
        self.log_dir = kwargs.setdefault('log_dir', config.log_dir)
        self.model_save_dir = os.path.join(self.log_dir, 'model')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        self.tb_logging = kwargs.get('tb',config.tb)
        if (self.tb_logging):
            self._tb_init()
        
        self.tot_epochs = 0
        self.episodes = 0
        self.episodes_rewards = deque(maxlen=100)

        self.policy_optimization_epochs = kwargs.get('policy_optimization_epochs', config.policy_optimization_epochs)
        self.policy_stopping_kl = kwargs.get('policy_stopping_kl', config.policy_stopping_kl)
        self.policy_sampling_ratio = 0.2
        
        self.policy_clip_range = kwargs.get('policy_clip_range', config.policy_clip_range)
        self.gae_lambda = kwargs.get('gae_lambda', config.gae_lambda)
        self.batch_size = kwargs.get('memory_batch_size', config.memory_batch_size)
        self.entropy_coeff = kwargs.get('entropy_beta', config.entropy_beta)
        self.vf_coeff = kwargs.get('vf_coeff', config.vf_coeff)
        self.policy_gradient_clip = float('inf')
        # tb tracking fields
        self.tb_loss_value = 0.0
        self.tb_loss_policy = 0.0
        self.tb_loss_entropy = 0.0
        self.tb_avg_return = 0.0
        self.tb_traj_mean_value = 0.0
        self.tb_policies_distance = 0.0
        self.tb_avg_entropy = 0.0
        self.tb_epi = 0 # epochs per iteration

    
    def collect_trajectories(self, num_trajectories):
        """collect trajectories over a horizon from environment"""
        agent_scores = np.zeros(self.num_agents)
        horizon = deque()
        env = self.env
        env_info = env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
        for _ in range(num_trajectories):
            values, actions, log_probs = self.policy.np_action(states, eps= 0.0)
            # send all actions to tne environment and collect observation
            env_info = env.step(actions)[self.brain_name]     
            next_states = np.array(env_info.vector_observations)
            rewards = np.array(env_info.rewards)                      
            dones = np.array(env_info.local_done)
            if (np.any(np.isnan(next_states)) or np.any(np.isnan(rewards)) or np.any(np.isnan(dones))) == False:
                # update the score (for each agent)
                agent_scores += rewards                           
                # check if any agents finished an episode
                for i,d in enumerate(dones):
                    if d:
                        self.episodes_rewards.append(agent_scores[i])
                        agent_scores[i] = 0.0
                        self.episodes += 1
                horizon.append((states,actions,rewards,dones,log_probs,values))
                states = next_states
        
        # process horizon backwards for generalized advantage
        next_values = self.policy.get_np_value(states)
        next_v = torch.tensor(next_values, dtype=torch.float32)
        gae = torch.zeros(self.num_agents, 1).to(self.device)
        trajectory = [None]*len(horizon)
        i = 0
        while (len(horizon) > 0):
            states,actions,rewards,dones,log_probs,values = horizon.pop()
            # turn np arrays into tensors
            states, actions, rewards, dones, log_probs,values = map(
                lambda x: torch.tensor(x).float().to(self.device),
                (states, actions, rewards, dones, log_probs, values)
            )
            goes_on = (1-dones).unsqueeze(-1)
            rewards = rewards.unsqueeze(-1)
            delta = rewards + self.gamma*next_v*goes_on - values
            gae = delta + self.gae_lambda*self.gamma*goes_on*gae
            returns = gae + values 

            # store in trajectory list
            trajectory[i] = (states, actions, log_probs, returns, gae, values)
            i += 1
            next_v = values
        
        # pack trajectories
        states, actions, old_log_probs, returns, gae, values = map(
            lambda x: torch.cat(x, dim=0), zip(*trajectory)
            )
        # calculate and normalize advantage
        adv = returns - values
        adv = (adv - adv.mean())  / (adv.std() + 1.0e-6)
        experiences_dict = {
            'states': states,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'returns': returns,
            'advantages': adv,
            'state_values': values
        }
        return experiences_dict

    def train(self,**kwargs):
        self.training_iterations = kwargs.pop('training_iterations', config.training_iterations)
        max_frames = kwargs.setdefault('trajectories',config.trajectories)
        self.training_iterations += self.saved_iteration  # in case we are continuing training
        env = self.env
        agent_scores = np.zeros(self.num_agents)         # initialize the score (for each agent)
        total_progress = tqdm(
                             total=self.training_iterations-self.iteration,
                             desc='step {}, epochs: {}, episodes: {}, mean score:{:3.2f},'.format(
                                                    self.iteration, self.tot_epochs, self.episodes,self.tb_avg_return))
        
        prev_iteration = self.iteration
        while self.iteration < self.training_iterations:
            observations = self.collect_trajectories(max_frames)
            epochs, policy_loss, value_loss, entropy_loss, kl_distance, values_mean, entropy_mean = self._learn_step(observations)
            # store values for tensorboard
            self.tb_avg_return = np.mean(self.episodes_rewards)
            self.tb_loss_value = value_loss
            self.tb_loss_policy = policy_loss
            self.tb_loss_entropy = entropy_loss
            self.tb_traj_mean_value = values_mean
            self.tb_policies_distance = kl_distance
            self.tb_avg_entropy = entropy_mean
            self.tb_epi = epochs # epochs per iteration
            # update progress bar
            if (self.iteration+1) % 1 == 0:
                total_progress.update(self.iteration-prev_iteration)
                total_progress.set_description('step {:4d}, epochs: {:4d}, episodes: {:4d}, mean score:{:3.2f},'.format(
                                                    self.iteration, self.tot_epochs, self.episodes, self.tb_avg_return ))
                prev_iteration = self.iteration
            self._next_iter()

    def play(self, 
            episodes=10,
            **kwargs):
        """plays an environment with model for requested episodes."""
        logging.info('Starting a Play session')
        logging.info('Will run for %d episodes', episodes)
        agent_scores = np.zeros(self.num_agents)
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        while self.episodes < episodes:
            # initialize the score (for each agent)
            frames = 0            
            states = env_info.vector_observations
            actions = self.policy.np_deterministic_action(states) # self.eps)
            # send all actions to tne environment and collect observation
            env_info = self.env.step(actions)[self.brain_name]     
            next_states = np.array(env_info.vector_observations)
            rewards = np.array(env_info.rewards)                      
            dones = np.array(env_info.local_done)
            frames += 1
            agent_scores += rewards
            for i,d in enumerate(dones):
                if d:
                    self.episodes_rewards.append(agent_scores[i])
                    self.episodes += 1
                    print('Agent {} Episode {}\t \t Score: {:.2f}'
                    .format(
                        i,
                        self.episodes,
                        agent_scores[i]
                    ))
                    agent_scores[i] = 0.0
            states = next_states
                
        print('Total Episodes: {}\t running mean score: {:.2f}'
            .format(
                self.episodes,
                np.mean(self.episodes_rewards)))

    def _learn_step(self, observations_dict):
        """perform one PPO learning step across defined epochs"""
        all_states = observations_dict['states']
        all_actions = observations_dict['actions']
        all_old_log_probs = observations_dict['old_log_probs']
        all_returns = observations_dict['returns']
        all_gae = observations_dict['advantages']
        all_values = observations_dict['state_values']
        n_samples = len(all_returns)
        for policy_epochs in range(self.policy_optimization_epochs):
            idx = np.random.choice(n_samples, self.batch_size, replace=False)
            self.tot_epochs += 1
            # sample a batch
            with torch.no_grad():
                states = all_states[idx]
                actions = all_actions[idx]
                old_log_probs = all_old_log_probs[idx]
                returns = all_returns[idx]
                values = all_values[idx]
                gae = all_gae[idx]
            # get new predictions   
            values_pred, log_probs, entropy = self.policy.get_probs_and_value(states,actions)
            # compute value loss
            value_loss = F.smooth_l1_loss(values_pred, returns)
            # compute policy loss
            ratio = (log_probs - old_log_probs).exp()
            policy_objective = gae*ratio
            policy_objective_clamped = torch.where(gae > 0, (1+self.policy_clip_range) * gae, (1-self.policy_clip_range) * gae)
            policy_loss = -torch.min(policy_objective, policy_objective_clamped).mean()
            # compute entropy loss
            entropy_loss = -entropy.mean()
            # objective
            ppo_objective = policy_loss + self.vf_coeff*value_loss + self.entropy_coeff*entropy_loss
            # optimize PPO DNN
            self.policy_optimizer.zero_grad()
            ppo_objective.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.policy_gradient_clip)
            self.policy_optimizer.step()
            # check if we are deviating too much from old policy and stop early if needed
            with torch.no_grad():
                values, log_probs, _ = self.policy.get_probs_and_value(all_states,all_actions)
                kl = (all_old_log_probs - log_probs).mean()
                if kl > self.policy_stopping_kl: 
                    break
        return policy_epochs, policy_loss, value_loss, entropy_loss, kl.item(), values.mean(), entropy.mean()

    def save_model(self, path=None):
        """Saves the model."""
        data = {
            'iteration': self.iteration,
            'policy_state_dict': self.policy.state_dict(),
            'policy_optim_state_dict': self.policy_optimizer.state_dict(),
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
            self.policy.train()
            logging.info('Loaded model: {}'.format(load_path))
            logging.info('iteration: {}'.format(self.iteration))

    def _next_iter(self):
        """increases training iterator and performs logging/model saving"""
        self.iteration += 1
        
        if (self.iteration+1) % self.model_save_period ==0:
            self.save_model(os.path.join(self.model_save_dir, 'model_{:4d}.pt'.format(self.save_counter)))
            self.save_counter += 1
        if self.tb_logging and ((self.iteration+1) % self.tensorboard_update_period ==0):
            # save latest model
            self.save_model(os.path.join(self.model_save_dir, 'model_latest.pt'))
            # write to tb log
            self._tb_write()
        # decrease LR
        self.policy_scheduler.step()
        return self.iteration

    def _tb_init(self):
        """initialize tensorboard logging"""

        ts = datetime.datetime.now().replace(microsecond=0).strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'tb',ts))
        self.tensorboard_update_period = 1
        return

    def _tb_write(self):
        """Writes training data to tensorboard summary writer. Should be overloaded by sub-classes."""
        it = self.iteration
        # tb tracking fields
        self.writer.add_scalar('loss/policy', self.tb_loss_policy, it)
        self.writer.add_scalar('loss/value', self.tb_loss_value, it)
        self.writer.add_scalar('loss/entropy', self.tb_loss_entropy, it)

        self.writer.add_scalar('mean/ return', self.tb_avg_return, it)
        self.writer.add_scalar('mean/entropy', self.tb_avg_entropy, it)
        self.writer.add_scalar('policies kl distance', self.tb_policies_distance, it)
        self.writer.add_scalar('mean/trajectories value', self.tb_traj_mean_value, it)
        self.writer.add_scalar('epochs', self.tb_epi, it)
        self.writer.flush()