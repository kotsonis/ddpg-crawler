#ref: daisatojp github

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging
from absl import flags
from networks import Actor, Critic
from networks import hidden_init
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

config = flags.FLAGS

class Actor(Actor):
    """
    Distributional Actor.

    The forward pass outputs a vector of means and covariance matrix (optionally)
    which means that we can sample an action (based on the distribution), instead
    of being deterministic and greedy.

    """
    def __init__(self, **kwargs):
        super(DistribActor, self).__init__(**kwargs)
        self.lin1 = nn.Linear(self.ds, 256)
        self.lin2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, self.da)
        self.cholesky_layer = nn.Linear(256, (self.da * (self.da + 1)) // 2)

    def forward(self, state):
        """
        forwards input through the network
        :param state: (B, ds)
        :return: mean vector (B, da) and cholesky factorization of covariance matrix (B, da, da)
        """
        device = state.device
        B = state.size(0)
        ds = self.ds
        da = self.da
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        mean = torch.sigmoid(self.mean_layer(x))  # (B, da)
        mean = -1.0 + (2.0) * mean
        cholesky_vector = self.cholesky_layer(x)  # (B, (da*(da+1))//2)
        cholesky_diag_index = torch.arange(da, dtype=torch.long) + 1
        cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        cholesky_vector[:, cholesky_diag_index] = F.softplus(cholesky_vector[:, cholesky_diag_index])
        tril_indices = torch.tril_indices(row=da, col=da, offset=0)
        cholesky = torch.zeros(size=(B, da, da), dtype=torch.float32).to(device)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
        return mean, cholesky

    def action(self, state):
        """
        :param state: (ds,)
        :return: an action
        """
        with torch.no_grad():
            mean, cholesky = self.forward(state)
            action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
            action_distribution.rsample
            actions = action_distribution.sample()
        return actions

class CriticContinuous(Critic):
    """
    :param env: OpenAI gym environment
    """
    def __init__(self, state_size, action_size):
        super(CriticContinuous, self).__init__()
        self.ds = state_size
        self.da = action_size
        self.lin1 = nn.Linear(self.ds + self.da, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 1)

    def forward(self, state, action):
        """
        :param state: (B, ds)
        :param action: (B, da)
        :return: Q-value
        """
        h = torch.cat([state, action], dim=1)  # (B, ds+da)
        h = F.relu(self.lin1(h))  # (B, 128)
        h = F.relu(self.lin2(h))  # (B, 128)
        v = self.lin3(h)  # (B, 1)
        return v