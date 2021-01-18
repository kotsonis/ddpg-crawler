import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging
from absl import flags
from networks import models
from networks.models import hidden_init

config = flags.FLAGS
class SDPGActor(models.Actor):
    def __init__(self, **kwargs):
        """initialize a DDPG actor (no changes for SPDG on actor)"""
        super(SDPGActor,self).__init__(**kwargs)
        return

class SDPGCritic(models.Critic):
    def __init__(self, **kwargs):
        super(SDPGCritic, self).__init__(**kwargs)
        
        return
    def _set_networks(self):
        self.state_fc_1 = nn.Linear(self.ds, self.dim_dense_1).to(self.device)
        self.bn1 = nn.BatchNorm1d(self.dim_dense_1).to(self.device)
        self.state_fc_2 = nn.Linear(self.dim_dense_1,self.dim_dense_2).to(self.device)
        self.action_fc = nn.Linear(self.da, self.dim_dense_2).to(self.device)
        self.sample_fc = nn.Linear(1,self.dim_dense_2).to(self.device)
        self.dense3 = nn.Linear(self.dim_dense_2, self.dim_dense_3).to(self.device)
        self.output_fc = nn.Linear(self.dim_dense_3,1).to(self.device)
        return

    def _reset_parameters(self):
        self.state_fc_1.weight.data.uniform_(*hidden_init(self.state_fc_1))
        self.state_fc_2.weight.data.uniform_(*hidden_init(self.state_fc_2))
        self.action_fc.weight.data.uniform_(*hidden_init(self.action_fc))
        self.sample_fc.weight.data.uniform_(*hidden_init(self.sample_fc))
        self.dense3.weight.data.uniform_(*hidden_init(self.dense3))
        self.output_fc.weight.data.uniform_(*hidden_init(self.output_fc))
        return
    
    
    def forward(self, states, actions, noise_samples):
        """calculates Q values for given samples.
        
        It's expected that noise is sampled from Gaussian distribution.
        The mean Q value per (state,action) pair is stored in self.Q """
        # bit complex, so I have put the resulting tensor dimensions in comments
        # states=[M,agents,ds]
        # actions=[M,agents,da]
        # noise_samples = [M*agents,train_samples,1]
        s = states.view(-1,self.ds)                 # [M*agents, ds]
        B = s.shape[0] # B=M*agents
        # process states
        xs = self.bn1(self.state_fc_1(s))           # [B,dnn_1_dim]
        xs = F.leaky_relu(xs)                       # [B,dnn_1_dim]
        xs = self.state_fc_2(xs).unsqueeze_(1)       # [B,1,dnn_2_dim]
        # process actions
        a = actions.view(-1,self.da)                # [B,da]
        xa = self.action_fc(a).unsqueeze_(1)        # [B,1,dnn_2_dim]
        # process samples
        atoms = noise_samples.shape[-2]
        x_s = noise_samples.view(-1,1)              # [B*train_samples, 1]
        x_s = (self.sample_fc(x_s)                  # [B*train_samples,dnn_2_dim]
             .view(B,-1,self.dim_dense_2))          # [B,train_samples,dnn_2_dim]
        #x_s now (B*num_samples,num_atoms,dim_dense_2)
        x = (F.leaky_relu(self.dense3(xs+xa+x_s))    # [B,train_samples,dnn_2_dim]
            .view(-1,self.dim_dense_3))              # [B*train_samples,dnn_2_dim]
        x = self.output_fc(x)                       # [B*train_samples,1]
        output_samples = x.view(B,-1)               # [B, train_samples]
        self.Q_val = output_samples.mean(1)         # [B]
        return output_samples