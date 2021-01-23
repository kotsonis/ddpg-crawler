""" DDPG Agents utilizing n-step rewards """

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

class nstepDDPG(agents.base.Agent):
    """DDPG Agent using n-step rewards and prioritized buffer"""
    def __init__(self,**kwargs):
        """Initializes Agent, with an n-step prioritized buffer

        Params:
        - n_step : number of steps to lookahead rewards
        """
        ## create base Agent with the new buffer
        kwargs['replay_buffer_class']= replay.NStepPriorityReplay
        super(nstepDDPG,self).__init__(**kwargs)

    ## override parent learn step function to use n-steps
    def _learn_step(self):
        """performs one learning step"""
        # ---------------- sample a batch of experiences ---------------------- #
        states,actions,rewards,next_states,dones, gammas, weights, indices = self.memory.sample()
        # decay beta for next sampling (Prioritized Experience Replay related)
        self.memory.decay_beta()
        
        # unravel S,A,R,S,done in case we have stored a multi-agent experience
        B = states.shape[0]
        states = states.view(-1,self.ds)
        actions = actions.view(-1,self.da)
        rewards = rewards.view(-1,1)
        next_states = next_states.view(-1,self.ds)
        dones = dones.view(-1,1)
        gammas = gammas.view(-1,1)
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


