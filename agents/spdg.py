# distributional deterministic policy gradient based on samplings
# classes provided are:
# SDPG - base, with simple replay buffer
# SDPG_per - base + prioritized experience replay
# SDPG_n - base + n_step returns replay buffer
# SDPG_n_per - SDPG_per + n_step returns

import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import copy

import utils
import networks
from agents.nstep import nstepDDPG
from agents.base import Agent
from absl import logging
from absl import flags

flags.DEFINE_integer(name='num_atoms',default=52,
                  help='number of atoms to sample for the critic Q_value distribution')
config = flags.FLAGS

class SDPGAgent(Agent):
    def __init__(self,**kwargs):
      config = flags.FLAGS
      # ----------------- create online & target actors -------------------- #
      actor_dnn_class = kwargs.setdefault('actor_dnn_class',networks.SDPGActor)
      critic_dnn_class = kwargs.setdefault('critic_dnn_class',networks.SDPGCritic)
      self.num_atoms = kwargs.setdefault('num_atoms',config.num_atoms)
      super(SDPGAgent,self).__init__(**kwargs)


    
    def _learn_step(self):
        """ samples a batch of experiences to perform one step of learning Actor/Critic """

        num_atoms = self.num_atoms
        # ---------------- sample a batch of experiences ---------------------- #
        states,actions,rewards,next_states,dones,gammas, weights, indices = self.sample()
        
        # in case we have stored a multi-agent experience, we unroll the S,A,R,S,done, gammas
        
        states = states.view(-1,self.ds)
        self.B = states.shape[0]
        actions = actions.view(-1,self.da)
        rewards = rewards.view(-1,1)
        next_states = next_states.view(-1,self.ds)
        dones = dones.view(-1,1)
        gammas = gammas.view(-1,1)
        
        # -------------------- get target Q value ----------------------------- #
        with torch.no_grad():
            future_action = self.target_actor(next_states) # size [B,da]
            # sample from gaussian distribution for future_q sampling
            noise_target = torch.normal(
                                        mean=0.5,
                                        std=1,
                                        size= (self.B, num_atoms, 1),
                                        device=self.device,
                                        requires_grad=False) # size [B,num_atoms,1]
            q_next_state = self.target_critic(next_states, future_action, noise_target) # size [B,num_atoms]
            # future value is 0 if episode is done
            q_next_state *= (1-dones) 
            q_target = rewards + gammas*q_next_state #[B, num_atoms]

        noise_online = torch.normal(
                                    mean=0.5,
                                    std=1,
                                    size=(self.B, num_atoms, 1),
                                    device=self.device,
                                    requires_grad=False) # size [B,num_atoms,1]
        q_online = self.critic(states,actions, noise_online) # [B,num_atoms]
        
        # store for tensorboard and priority calculation
        with torch.no_grad():
            self._current_q = q_online.detach().mean(1)
            td_error = q_target.mean(1) - self._current_q
            self.mean_est_q = self._current_q.mean()
        
        # calculate wasserstein distance between q_target and q_online
        # sort ascending on num_atoms dimension (ie largest Q contributor at top of each B)
        q_target, _ = torch.sort(q_target, dim=1)
        q_online, _ = torch.sort(q_online, dim=1)
        
        #expand q_target and q_online to allow for distance across each atom contribution
        q_target_tile = q_target.unsqueeze(-2).repeat(1,num_atoms,1) # [B,num_atoms,num_atoms]
        q_online_tile = q_online.unsqueeze(-1).repeat(1,1,num_atoms) # [B,num_atoms,num_atoms]
        # example q_target [0][0] = y, q_online[0][0]=x
        #    q_target_tile [0][0 .. num_atoms][0] = y
        #    q_online_tile [0][0][0 .. num_atoms] = x

        error_loss = q_online_tile - q_target_tile  # x_tilde - y_tilde
        # user huber loss as a surrogate for the wasserstein distance (L)
        huber_loss = F.smooth_l1_loss(q_online_tile,q_target_tile,reduction='none')
        
        # create vector of importance taus, ie the tau in (tau-dirac((x-y)<0.)
        min_tau = 1/(2*num_atoms)
        max_tau = (2*num_atoms+1)/(2*num_atoms)
        tau = torch.arange(start=min_tau, end=max_tau, step= 1/num_atoms, device=self.device, requires_grad=False).unsqueeze(0)
        inv_tau = 1.0 - tau

        loss = torch.where(error_loss.lt(0.0), inv_tau * huber_loss, tau*huber_loss) # loss size [B, num_atoms, num_atoms]
        loss = loss.mean(dim=2)                 # size [B, num_atoms]
        loss = (loss.mean(dim=1)                 # size [B]
                   .view(self.sample_size,-1))   # size [M, agents]
        critic_loss = (weights*loss)            # size [M, agents]
        critic_loss_batch = critic_loss.mean() # size scalar 
        
        # store loss for tensorboard
        with torch.no_grad():
            self.mean_loss_q = critic_loss_batch.detach()

        # --------------------- optimize the critic ---------------------------- #
        self.critic_optimizer.zero_grad()
        critic_loss_batch.backward()
        self.critic_optimizer.step()
        # ---------------------- compute actor loss ---------------------------- #
        # we want maximum reward (Q) so loss is negative of current Q
        noise_actor  = torch.normal(
                                    mean=0.5,
                                    std=1,
                                    size=(self.B, num_atoms, 1),
                                    device=self.device,
                                    requires_grad=False) # size [B,num_atoms,1]

        actor_loss = - (self.critic(states, self.actor(states), noise_actor) # [M*Agents,num_atoms]
                                  .mean()) # scalar
                                   
        # store loss for tensorboard
        with torch.no_grad():
            self.mean_loss_p = actor_loss                          
        # --------------------- optimize the actor ----------------------------- #
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # --------------------- Prioritized Replay Buffer ---------------------- #
        
        # -------------- update priorities in the replay buffer ---------------- #
        # calculate TD_error
        # we do not want this to enter into computation graph for autograd
        if (self.hasPER):
            with torch.no_grad():
                
                td_error =  td_error.view(self.sample_size,-1).mean(dim=1)
                td_error = td_error.abs()
                self.min_td = td_error.min()
                #self.mean_td = td_error.mean()
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
        
    
