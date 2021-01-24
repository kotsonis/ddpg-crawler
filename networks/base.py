import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging
from absl import flags
flags.DEFINE_integer(name='actor_dim_dense_1',default=256,
                     help='output dimension of actor 1st hidden layer')
flags.DEFINE_integer(name='actor_dim_dense_2',default=256,
                     help='output dimension of actor 2nd hidden layer')
flags.DEFINE_integer(name='actor_dim_dense_3',default=256,
                     help='output dimension of actor 3nd hidden layer')
flags.DEFINE_integer(name='critic_dim_dense_1',default=256,
                     help='output dimension of critic 1st hidden layer')
flags.DEFINE_integer(name='critic_dim_dense_2',default=256,
                     help='output dimension of critic 2nd hidden layer')
flags.DEFINE_integer(name='critic_dim_dense_3',default=256,
                     help='output dimension of critic 3nd hidden layer')

config = flags.FLAGS
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, **kwargs):
        super(Actor, self).__init__()
        self.ds = kwargs['state_size']
        self.da = kwargs['action_size']
        self.dim_dense_1 = (kwargs
                           .setdefault(
                                     'actor_dim_dense_1',
                                     config.actor_dim_dense_1))
        self.dim_dense_2 = (kwargs
                           .setdefault(
                                     'actor_dim_dense_2',
                                     config.actor_dim_dense_2))
        self.dim_dense_3 = (kwargs
                           .setdefault(
                                     'actor_dim_dense_3',
                                     config.actor_dim_dense_3))
        self.device = kwargs.get('device','cpu')

        self.state_fc_1 = nn.Linear(self.ds, self.dim_dense_1).to(self.device)
        self.bn1 = nn.BatchNorm1d(self.dim_dense_1).to(self.device)
        self.hidden_fc_1 = nn.Linear(self.dim_dense_1,self.dim_dense_2).to(self.device)
        self.hidden_fc_2 = nn.Linear(self.dim_dense_2, self.dim_dense_3).to(self.device)
        self.output_fc = nn.Linear(self.dim_dense_3, self.da).to(self.device)
        self.reset_parameters()
        return
    
    def reset_parameters(self):
        self.state_fc_1.weight.data.uniform_(*hidden_init(self.state_fc_1))
        self.hidden_fc_1.weight.data.uniform_(*hidden_init(self.hidden_fc_1))
        self.hidden_fc_2.weight.data.uniform_(*hidden_init(self.hidden_fc_2))
        self.output_fc.weight.data.uniform_(*hidden_init(self.output_fc))

    def forward(self, states):
        xs = states.view(-1,self.ds)
        x = F.leaky_relu(self.bn1(self.state_fc_1(xs)))
        x = F.leaky_relu(self.hidden_fc_1(x))
        x = F.leaky_relu(self.hidden_fc_2(x))
        return torch.tanh(self.output_fc(x))
    def action(self, state):
        """ Generates actions from states. """
        with torch.no_grad():
            actions = self.forward(state)
        return actions


class Critic(nn.Module):
    def __init__(self, **kwargs):
      super(Critic, self).__init__()
      self.ds = kwargs['state_size']
      self.da = kwargs['action_size']
      self.device = kwargs.get('device','cpu')
      self.dim_dense_1 = (kwargs
                           .setdefault(
                                     'critic_dim_dense_1',
                                     config.critic_dim_dense_1))
      self.dim_dense_2 = (kwargs
                           .setdefault(
                                     'critic_dim_dense_2',
                                     config.critic_dim_dense_2))
      self.dim_dense_3 = (kwargs
                           .setdefault(
                                     'critic_dim_dense_3',
                                     config.critic_dim_dense_3))
      self._set_networks()
      self._reset_parameters()
      return
    def _set_networks(self):
        self.state_fc_1 = nn.Linear(self.ds, self.dim_dense_1)
        self.bn1 = nn.BatchNorm1d(self.dim_dense_1)
        self.state_fc_2 = nn.Linear(self.dim_dense_1,self.dim_dense_2)
        self.hidden_fc_1 = nn.Linear(self.dim_dense_2+self.da, self.dim_dense_2)
        self.hidden_fc_2 = nn.Linear(self.dim_dense_2,self.dim_dense_3)
        self.output_fc = nn.Linear(self.dim_dense_3,1)
        return
    def _reset_parameters(self):
        self.state_fc_1.weight.data.uniform_(*hidden_init(self.state_fc_1))
        self.state_fc_2.weight.data.uniform_(*hidden_init(self.state_fc_2))
        self.hidden_fc_1.weight.data.uniform_(*hidden_init(self.hidden_fc_1))
        self.hidden_fc_2.weight.data.uniform_(*hidden_init(self.hidden_fc_2))
        
        self.output_fc.weight.data.uniform_(*hidden_init(self.output_fc))

    def forward(self, states, action):
        xs = F.leaky_relu(self.bn1(self.state_fc_1(states)))
        xs = F.leaky_relu(self.state_fc_2(xs))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.hidden_fc_1(x))
        x = F.leaky_relu(self.hidden_fc_2(x))
        return self.output_fc(x)
