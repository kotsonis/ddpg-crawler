import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import copy

from models import Actor_SDPG, Critic_SDPG
from buffers import PrioritizedReplayBuffer,nStepPER
from config import Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tile(a, dim, n_tile):
    """ helper function: tile of torch tensor... since we are not using latest pytorch"""
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)

class DPG():
    def __init__(self, config):
        # ----------------- create online & target actors -------------------- #
        self.actor = Actor_SDPG(config.state_size, config.action_size, dense1_size=config.dense1_size, dense2_size=config.dense2_size).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        # ---------------- create online & target critics -------------------- #
        self.critic = Critic_SDPG(config.state_size, config.action_size,num_atoms = config.num_atoms, dense1_size=config.dense1_size, dense2_size=config.dense2_size).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        # --------------------- store hyperparameters ------------------------ #
        # Environment
        self.state_size = config.state_size
        self.action_size = config.action_size
        
        # discounting and discounting steps
        self.discount = config.gamma
        self.n_step = config.n_step
        
        # Prioritized experience replay memory settings
        self.PER_alpha = config.PER_alpha
        self.PER_buffer = config.PER_buffer
        self.batch_size = config.PER_batch_size
        self.PER_eps = config.PER_eps # minimum priority when updating experiences
        self.beta = config.PER_beta_start
        self.beta_step = config.PER_beta_decay
        self.beta_max = config.PER_beta_max

        # training parameters : Soft update rate / exploration / Update every
        self.tau = config.soft_update_tau
        self.eps = config.eps_start
        self.eps_decay = config.eps_decay_rate
        self.eps_min = config.epsilon_min
        self.update_every = config.update_every
        self.t_step = 0
        self.train_step = 0
        self.num_atoms = config.num_atoms
        # Noise process
        self.noise = OUNoise(self.action_size)

        # initialize the replay buffer
        self.memory = nStepPER(size=self.PER_buffer,batch_size=self.batch_size,alpha=self.PER_alpha,n_step=self.n_step,gamma=self.discount)

        
    def save_models(self, save_dir):
        """ Save actor & critic models to directory """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.critic.state_dict(),
            os.path.join(save_dir, 'critic_online_net.pth'))
        torch.save(
            self.critic_target.state_dict(),
            os.path.join(save_dir, 'critic_target_net.pth'))
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor_online_net.pth'))
        torch.save(
            self.actor_target.state_dict(),
            os.path.join(save_dir, 'actor_target_net.pth'))


    def load_models(self, save_dir):
        """ Load actor & critic models from directory """
        self.critic.load_state_dict(torch.load(
            os.path.join(save_dir, 'critic_online_net.pth')))
        self.critic_target.load_state_dict(torch.load(
            os.path.join(save_dir, 'critic_target_net.pth')))
        self.actor.load_state_dict(torch.load(
            os.path.join(save_dir, 'actor_online_net.pth')))
        self.actor_target.load_state_dict(torch.load(
            os.path.join(save_dir, 'actor_target_net.pth')))
        

    def step(self, state, action, reward, next_state, done):
        """Process one step of Agent/environment interaction"""

        # add frame/maximum frames to reward
        
        # Save experience / reward
        self.memory.add(state,action,reward,next_state,done)
        self.t_step += 1
         # Learn every UPDATE_EVERY time steps.
        if (self.t_step < 2000) : return self.train_step
        if (self.t_step + 1) % self.update_every == 0:
            # If enough samples are available in memory, perform a learning step
            if len(self.memory) > self.batch_size:
                self.learn()
        return self.train_step

    def act(self,states, add_noise=True, t=1):
        """ act according to target policy based on state """

        # move states into torch tensor on device
        state = torch.FloatTensor(states).to(device)
        # turn off training mode
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy()
            # if we are being stochastic, add noise weighted by exploration
            if add_noise:
                action = self.noise.get_action(action,t)
        self.actor_target.train()

        return np.clip(action, -1, 1)  # TODO: clip according to Unity environment feedback

    def learn(self):
        """ samples a batch of experiences to perform one step of learning Actor/Critic """

        num_atoms = self.num_atoms
        self.train_step += 1 # increase training step
        # ---------------- sample a batch of experiences ---------------------- #
        states,actions,rewards,next_states,dones,gammas, weights, indices = self.memory.sample(self.batch_size, beta=self.beta)
        
        # in case we have stored a multi-agent experience, we unroll the S,A,R,S,done, gammas
        batch_size = states.shape[0]
        states = states.view(-1,self.state_size)
        actions = actions.view(-1,self.action_size)
        rewards = rewards.view(-1,1)
        next_states = next_states.view(-1,self.state_size)
        dones = dones.view(-1,1)
        gammas = gammas.view(-1,1)
        
        # decay beta for next sampling (Prioritized Experience Replay related)
        self.beta = min(self.beta+self.beta_step, self.beta_max)
        
        # -------------------- get target Q value ----------------------------- #
        #with torch.no_grad():
        future_action = self.actor_target(next_states)
    
        # sample from gaussian distribution for future_q sampling
        # size : size=[train_params.BATCH_SIZE, train_params.NUM_ATOMS, train_params.NOISE_DIMS]
        q_sample_noise = torch.tensor(np.random.normal(size= (rewards.shape[0], num_atoms, 1)),dtype=torch.float).to(device)  #todo, add how many taus we want.. # SDPG specific
        q_hat_sample_noise = torch.tensor(np.random.normal(size= (rewards.shape[0], num_atoms, 1)),dtype=torch.float).to(device)  #todo, add how many taus we want.. # SDPG specific
        q_future_state = self.critic_target(next_states, future_action, q_sample_noise)     # SDPG specific
        # future value is 0 if episode is done
        q_future_state *= (1-dones) 
        q_target = rewards + gammas*q_future_state

        # def train_step(self, real_samples,  IS_weights, learn_rate, l2_lambda, num_atoms):
        # --------------=------ get current Q value ---------------------------- #
        q_online = self.critic(states,actions, q_hat_sample_noise)
        

        #q_target_sorted_indexes = torch.argsort(q_target, dim=-1)
        q_target_sorted, _ = torch.sort(q_target,dim=1)

        #q_online_sorted_indexes = torch.argsort(q_online, dim=-1)
        #q_online_sorted = torch.gather(q_online, index=q_online_sorted_indexes, dim=-1)
        q_online_sorted, _ = torch.sort(q_online, dim=1)
        #q_target_tile = torch.tile(q_target_sorted.unsqueeze(-2),(1,1,num_atoms))
        #q_online_tile = torch.tile(q_online_sorted.unsqueeze(-1),(1,num_atoms,1))
        q_target_tile = tile(q_target_sorted.unsqueeze(-2),-2,num_atoms)
        q_online_tile = tile(q_online_sorted.unsqueeze(-1),-1,num_atoms)

        error_loss = q_target_tile - q_online_tile
        huber_loss = F.smooth_l1_loss(q_target_tile, q_online_tile,reduction='none')
        
        min_tau = 1/(2*num_atoms)
        max_tau = (2*num_atoms+1)/(2*num_atoms)
        tau = torch.arange(start=min_tau, end=max_tau, step= 1/num_atoms, device=device).unsqueeze(0)
        inv_tau = 1.0 - tau
        #inv_huber = torch.matmul(inv_tau, huber_loss)
        #huber = torch.matmul(tau, huber_loss)
        loss = torch.where(error_loss.lt(0.0), inv_tau * huber_loss, tau*huber_loss)
        loss = loss.mean(dim=2)
        loss = loss.sum(dim=1)
        # loss = torch.sum(torch.mean(loss,dim=-1),dim=-1)
        critic_loss = (weights*loss.view(batch_size,-1))
        critic_loss_batch = critic_loss.mean()
        #critic_loss = critic_loss.mean()
        # --------------------- compute critic loss ---------------------------- #
        #  = mean(IS_weights * MSE(Q))    
        # critic_loss = (weights*((q_target-q_online)**2).view(-1,self.batch_size,1)).mean()
        # critic_loss = critic_loss.mean()
        # --------------------- optimize the critic ---------------------------- #
        self.critic_optimizer.zero_grad()
        critic_loss_batch.backward()
        self.critic_optimizer.step()
        # ---------------------- compute actor loss ---------------------------- #
        # we want maximum reward (Q) so loss is negative of current Q
        # noise = torch.tensor(np.random.normal(size= (rewards.shape[0], num_atoms, 1)),dtype=torch.float)  #todo, add how many taus we want.. # SDPG specific
        
        actor_loss = - self.critic(states, self.actor(states), q_hat_sample_noise).mean(dim=1).mean()
        #actor_loss = actor_loss.mean(dim=1)
        #actor_loss_batch = -actor_loss.mean()
        # --------------------- optimize the actor ----------------------------- #
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # --------------------- Prioritized Replay Buffer ---------------------- #
        # -------------- update priorities in the replay buffer ---------------- #
        # calculate TD_error
        # we do not want this to enter into computation graph for autograd
        with torch.no_grad():
            td_error = torch.mean(q_target, dim=1)
            td_error -= torch.mean(q_online, dim=1)
            td_error = td_error.abs().view(batch_size,-1)
            # td_error = error_loss.view(batch_size,-1)
            td_error = td_error.mean(dim=1)
        #    td_error = loss.view(batch_size,-1).mean(axis=1).abs()
            new_p = td_error + self.PER_eps
            
#            if (torch.any(torch.isnan(new_p))): 
#                print('priorities: got a NaN reward. Need to fix it. Priorities are : {}'.format(new_p))
            # -------------------- update PER priorities ----------------------- #
            self.memory.update_priorities(indices, new_p.cpu().data.numpy().tolist())
        
        # -------------------- soft update target networks --------------------- #
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)
        
        # ------------------- hard update target networks ---------------------- #
        if (((self.train_step +1) % 1000) == 0): 
            self.soft_update(self.critic, self.critic_target, 1.0)
            self.soft_update(self.actor, self.actor_target, 1.0)
            
        # ------------------- update noise and exploration --------------------- #
        self.eps *= self.eps_decay
        self.eps = max(self.eps, self.eps_min)
        
        return
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, max_sigma=0.3, min_sigma=0.00, decay_period=5000, high=1.0, low=-1.0):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.action_dim = size
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.low = low
        self.high = high
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        self.sigma = self.max_sigma

    def evolve_state(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t/self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
    

