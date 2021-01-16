#ref daisatojp

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.tensor


class CriticContinuous(nn.Module):
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
