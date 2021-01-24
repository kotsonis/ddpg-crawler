import numpy as np
import torch


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

    def __init__(self, action_dimension, device='gpu', scale=1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.device = device
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = torch.ones(self.action_dimension,dtype=torch.float,device=device) * self.mu
        self.reset()

    def reset(self):
        self.state = torch.ones(self.action_dimension,dtype=torch.float,device=self.device) * self.mu

    def sample(self):
        x = self.state
        noise = torch.randn(len(x),dtype=torch.float,device=self.device)
        dx = self.theta * (self.mu - x) + self.sigma * noise 
        self.state = x + dx
        return (self.state * self.scale)