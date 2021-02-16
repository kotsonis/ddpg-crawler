import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging
from absl import flags

config = flags.FLAGS

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class PPO(nn.Module):
    def __init__(self, 
                state_size, 
                action_size,
                action_bounds = (-0.9, 0.9),
                log_std_min=-20, 
                log_std_max=-0.25,
                hidden_dims=(256,256), 
                activation_fc=F.leaky_relu,
                
                **kwargs
                ):
        super(PPO,self).__init__()
        # was activation_fc=F.relu,
        self.ds = state_size
        self.da = action_size
        self.activation_fc = activation_fc
        self.env_min, self.env_max = action_bounds
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.input_layer = nn.Linear(state_size, hidden_dims[0])
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            # initialize layer with random vals
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
            self.hidden_layers.append(hidden_layer)
        self.batch_norm_layer = nn.BatchNorm1d(hidden_dims[-1])
        self.output_layer_mean = nn.Linear(hidden_dims[-1], action_size)
        self.output_layer_mean.weight.data.uniform_(*hidden_init(self.output_layer_mean))
        self.output_layer_log_std = nn.Linear(hidden_dims[-1], action_size)
        self.output_layer_value = nn.Linear(hidden_dims[-1], 1)
        self.output_layer_value.weight.data.uniform_(*hidden_init(self.output_layer_value))

        # move to GPU if available
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

        self.env_min = torch.tensor(self.env_min,
                                    device=self.device, 
                                    dtype=torch.float32)

        self.env_max = torch.tensor(self.env_max,
                                    device=self.device, 
                                    dtype=torch.float32)
        
        self.nn_min = torch.tanh(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = torch.tanh(torch.Tensor([float('inf')])).to(self.device)
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.env_max - self.env_min) / \
                                    (self.nn_max - self.nn_min) + self.env_min

        
    def _format(self, state):
        """cast state to torch tensor and unroll """
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.view(-1,self.ds)
        return x

    def forward(self, state):
        """returns pre-tanh action means and log of std deviation for each state"""
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        
        # calculate non-linear higher dimension representation of state
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        # calculate policy mean 
        x_mean = self.output_layer_mean(x)
        # calculate the log of the standard deviation and clamp within reasonable values
        x_log_std = self.output_layer_log_std(x)
        value = self.output_layer_value(x)
        return value, x_mean, x_log_std
    
    def full(self, state, epsilon = 1e-6):
        """returns sampled action, log_prob of action, and greedy action for state"""
        # get mean and std log for states
        value, mean, log_std = self.forward(state)
        # get a Normal distribution with those values
        policy = torch.distributions.Normal(mean, log_std.exp())
        # sample an action input
        pre_tanh_action = policy.rsample()
        # convert to -1 ... 1 value
        tanh_action = torch.tanh(pre_tanh_action)
        # rescale action to action bounds
        action = self.rescale_fn(tanh_action)
        # get log probability and rescale to action bounds
        log_prob = policy.log_prob(pre_tanh_action) - torch.log(
            (1-tanh_action.pow(2)).clamp(0,1) + epsilon)
        # multiply the probs of each action dimension (sum the log_probs)
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        return value, action, log_prob, self.rescale_fn(torch.tanh(mean))

    def np_action(self, state, eps=0.5, noise=False):
        """returns an action and log probs in numpy format for environment step"""
        
        if np.random.random() < eps:
            value, mean, log_std = self.forward(state)
            policy = torch.distributions.Normal(mean, log_std.exp())
            action = self.rescale_fn(torch.tanh(mean))
            log_prob = policy.log_prob(mean)
            log_prob = log_prob.sum(-1).unsqueeze(-1)
            
        else:
            value, action, log_prob, mean = self.full(state)
        action_np = action.detach().cpu().numpy()
        log_prob_np = log_prob.detach().cpu().numpy()
        value_np = value.detach().cpu().numpy()

        return value_np, action_np, log_prob_np

    def np_deterministic_action(self, state):
        """get deterministic action instead of sampling"""
        value, mean, log_std = self.forward(state)
        action = self.rescale_fn(torch.tanh(mean))
        action_np = action.detach().cpu().numpy()
        return action_np
        
    def get_probs_and_value(self, state, action):
        """returns log probs and entropy for choosing provided action at given state"""
        value, mean, log_std = self.forward(state)
        # get a Normal distribution with those values
        policy = torch.distributions.Normal(mean, log_std.exp())
        # convert action back to pre-tanh value
        pre_tanh_action = torch.atanh(action)

        log_prob = policy.log_prob(pre_tanh_action) - torch.log(
            (1-action.pow(2)).clamp(0,1) + 1e-6)

        log_prob = log_prob.sum(-1).unsqueeze(-1)
        entropy = policy.entropy().unsqueeze(-1)
        return value, log_prob, entropy
    
    def get_np_value(self, state):
        """get current value of state"""
        value, mean, log_std = self.forward(state)
        value_np = value.detach().cpu().numpy()
        return value_np
