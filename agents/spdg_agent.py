import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from collections import deque
import copy

from utils import buffers
from agents import nstep_agents
from agents.base_agent import tile
from networks import sdpg_models

from absl import logging
from absl import flags
config = flags.FLAGS
flags.DEFINE_integer(
                    name='SPDG_num_atoms',
                  default=64,
                  help='number of atoms to sample for the critic Q_value distribution')

class SPDG(nstep_agents.nstepDDPG):
    def __init__(
                self,
                **kwargs):
        # ----------------- create online & target actors -------------------- #
        actor_dnn_class = sdpg_models.SDPGActor
        critic_dnn_class = sdpg_models.SDPGCritic
        self.num_atoms = config.SPDG_num_atoms
        super(SPDG,self).__init__(
                                 actor_dnn_class = actor_dnn_class,
                                 critic_dnn_class = critic_dnn_class,
                                 num_atoms = self.num_atoms,
                                 **kwargs)

    def _learn_step(self):
        """ samples a batch of experiences to perform one step of learning Actor/Critic """

        num_atoms = self.num_atoms
        # ---------------- sample a batch of experiences ---------------------- #
        states,actions,rewards,next_states,dones,gammas, weights, indices = self.memory.sample()
        # decay beta for next sampling (Prioritized Experience Replay related)
        self.memory.decay_beta()
        # in case we have stored a multi-agent experience, we unroll the S,A,R,S,done, gammas
        batch_size = states.shape[0]
        states = states.view(-1,self.ds)
        B = states.shape[0]
        actions = actions.view(-1,self.da)
        rewards = rewards.view(-1,1)
        next_states = next_states.view(-1,self.ds)
        dones = dones.view(-1,1)
        gammas = gammas.view(-1,1)
        
        # -------------------- get target Q value ----------------------------- #
        #with torch.no_grad():
        future_action = self.target_actor(next_states).detach()
    
        # sample from gaussian distribution for future_q sampling
        # size : size=[train_params.BATCH_SIZE, train_params.NUM_ATOMS, train_params.NOISE_DIMS]
        target_q_noise = torch.tensor(np.random.normal(size= (B, num_atoms, 1)),dtype=torch.float).to(self.device)  #todo, add how many taus we want.. # SDPG specific
        q_noise = torch.tensor(np.random.normal(size= (B, num_atoms,1)),dtype=torch.float).to(self.device)  #todo, add how many taus we want.. # SDPG specific
        q_future_state = self.target_critic(next_states, future_action, target_q_noise).detach()     # SDPG specific
        # future value is 0 if episode is done
        q_future_state *= (1-dones) 
        q_target = rewards + gammas*q_future_state

        # def train_step(self, real_samples,  IS_weights, learn_rate, l2_lambda, num_atoms):
        # --------------=------ get current Q value ---------------------------- #
        q_online = self.critic(states,actions, q_noise)
        
        # store for tensorboard and priority calculation
        with torch.no_grad():
            current_q = q_online.detach().mean(1)
            self.mean_est_q = current_q.mean()

        #q_target_sorted_indexes = torch.argsort(q_target, dim=-1)
        q_target_sorted, _ = torch.sort(q_target,dim=1)

        #q_online_sorted_indexes = torch.argsort(q_online, dim=-1)
        #q_online_sorted = torch.gather(q_online, index=q_online_sorted_indexes, dim=-1)
        q_online_sorted, _ = torch.sort(q_online, dim=1)
        #q_target_tile = torch.tile(q_target_sorted.unsqueeze(-2),(1,1,num_atoms))
        #q_online_tile = torch.tile(q_online_sorted.unsqueeze(-1),(1,num_atoms,1))
        q_target_tile = tile(q_target_sorted.unsqueeze(-2),-2,num_atoms, self.device)
        q_online_tile = tile(q_online_sorted.unsqueeze(-1),-1,num_atoms, self.device)

        error_loss = q_target_tile - q_online_tile
        huber_loss = F.smooth_l1_loss(q_target_tile, q_online_tile,reduction='none')
        
        min_tau = 1/(2*num_atoms)
        max_tau = (2*num_atoms+1)/(2*num_atoms)
        tau = torch.arange(start=min_tau, end=max_tau, step= 1/num_atoms, device=self.device).unsqueeze(0)
        inv_tau = 1.0 - tau
        #inv_huber = torch.matmul(inv_tau, huber_loss)
        #huber = torch.matmul(tau, huber_loss)
        loss = torch.where(error_loss.lt(0.0), inv_tau * huber_loss, tau*huber_loss)
        loss = loss.mean(dim=2)
        loss = loss.sum(dim=1)
        # loss = torch.sum(torch.mean(loss,dim=-1),dim=-1)
        critic_loss = (weights*loss.view(batch_size,-1))
        critic_loss_batch = critic_loss.mean(1).mean()
        # store loss for tensorboard
        with torch.no_grad():
            self.mean_loss_q = critic_loss_batch.detach()
            td_error = critic_loss.detach()

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
        #samples noise
        noise = torch.tensor(np.random.normal(size= (B, num_atoms, 1)),dtype=torch.float).to(self.device)

        actor_loss = - (self.critic(states, self.actor(states), noise) # [M*Agents,num_samples]
                                  .mean(dim=1) # [M*Agents]
                                  .mean()) # scalar
                                   
        # store loss for tensorboard
        with torch.no_grad():
            self.mean_loss_p = actor_loss                          
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
            current_q = torch.clamp_min(current_q,1.0)
            td_error = td_error.view(B,-1)
            td_error =  td_error.abs()
            td_error = td_error/(current_q.abs()+1e-6)
            td_error = td_error.view(batch_size,-1).mean(dim=1)
            self.min_td = td_error.min()
            self.max_td = td_error.max()
            new_p = td_error.cpu().data.numpy().clip(0,5.0)
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
        
"""
class OUNoise:
    ""Ornstein-Uhlenbeck process.""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        ""Initialize parameters and noise process.""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.action_dims = size
        self.reset()

    def reset(self):
        ""Reset the internal state (= noise) to mean (mu).""
        self.state = copy.copy(self.mu)

    def sample(self):
        ""Update internal state and return it as a noise sample.""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dims)
        self.state = x + dx
        return self.state
"""