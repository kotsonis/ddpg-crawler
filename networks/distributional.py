import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging
from absl import flags
from networks import Actor, Critic
from networks import hidden_init

config = flags.FLAGS


class SDPGActor(Actor):
    def __init__(self, **kwargs):
        """initialize a DDPG actor (no changes for SPDG on actor)"""
        super(SDPGActor,self).__init__(**kwargs)
        self.state_fc_1 = nn.Linear(self.ds, self.dim_dense_1).to(device=self.device)
        self.hidden_fc_1 = nn.Linear(self.dim_dense_1,self.dim_dense_1).to(device=self.device)
        self.output_fc = nn.Linear(self.dim_dense_2, self.da).to(self.device)
        cov_size = (self.da * (self.da +1))// 2
        self.hidden_fc_2 = nn.Linear(self.dim_dense_2, cov_size).to(self.device)
        self.reset_parameters()
        return

    def forward(self, states, actions=None, explore=True):
        xs = states.view(-1,self.ds)
        B = xs.size(0)
        # calculate non-linear higher dimension representation of state
        x = F.relu(self.state_fc_1(xs))
        x = F.relu(self.hidden_fc_1(x))

        # calculate policy mean / covariance matrix (via Cholesky triangular L)
        mean = torch.tanh(self.output_fc(x))
        # calculate covariance lower triangular matrix
        cholesky_vector = self.hidden_fc_2(x)  # (B, (da*(da+1))//2)
        # calculate the indices in cholesky vector that correspond to the diagonal entries of L
        cholesky_diag_index = torch.arange(self.da, dtype=torch.long) + 1
        cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        # calculate std deviation of diagonal and update vector
        std = F.softplus(cholesky_vector[:, cholesky_diag_index])
        cholesky_vector[:, cholesky_diag_index] = std
        # create lower triangular matrix and put cholesky output in
        tril_indices = torch.tril_indices(row=self.da, col=self.da, offset=0)
        cholesky = torch.zeros(size=(B, self.da, self.da), dtype=torch.float32).to(self.device)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector

        # create resulting multivariate distribution
        dist = torch.distributions.MultivariateNormal(mean, scale_tril=cholesky)

        # if no action given, then sample one from this distribution
        if actions is None:
            if (explore):
                actions = dist.sample()
            else:
                actions = mean
        #actions = torch.clip(actions,-1.0,1.0)
        log_prob = dist.log_prob(actions)
        # for a Normal distribution across each action dim, add: log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return actions, log_prob

    def action(self, state, eps=0.5, noise=True):
        """ Generates actions from states. """
        actions = self.forward(state, explore=True) # [B,da]
        return actions

class SDPGCritic(Critic):
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
        # bit complex, so I have put the resulting tensor dimensions
        # states=[M,agents,ds]
        # actions=[M,agents,da]
        # noise_samples = [M*agents,num_atoms,1]
        s = states.view(-1, self.ds)                 # [M*agents, ds]
        B = s.shape[0]                               # B=M*agents
        # process states
        xs = self.bn1(self.state_fc_1(s))            # [B,dnn_1_dim]
        xs = F.leaky_relu(xs)                        # [B,dnn_1_dim]
        xs = self.state_fc_2(xs).unsqueeze_(1)       # [B,1,dnn_2_dim]
        # process actions
        a = actions.view(-1, self.da)                 # [B,da]
        xa = self.action_fc(a).unsqueeze_(1)         # [B,1,dnn_2_dim]
        # process samples
        # atoms = noise_samples.shape[-2]
        x_s = noise_samples.view(-1, 1)             # [B*num_atoms, 1]
        x_s = (self.sample_fc(x_s)                  # [B*num_atoms,dnn_2_dim]
              .view(B, -1, self.dim_dense_2))          # [B,num_atoms,dnn_2_dim]
        # x_s now (B*num_samples,num_atoms,dim_dense_2)
        x = (F.leaky_relu(self.dense3(xs+xa+x_s))    # [B,num_atoms,dnn_2_dim]
            .view(-1, self.dim_dense_3))              # [B*num_atoms,dnn_2_dim]
        out = self.output_fc(x)                       # [B*num_atoms,1]
        output_samples = out.view(B, -1)               # [B, num_atoms]

        # Q value is just the average over num_atoms] ie output_samples.mean(1) [B]
        return output_samples

class ValuePPO(nn.Module):
    def __init__(self,
                state_size,
                hidden_dims=(256,256), 
                activation_fc=F.leaky_relu,
                **kwargs):
        super(ValuePPO, self).__init__()
        # was activation_fc=F.relu,
        self.ds = state_size
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(state_size, hidden_dims[0])
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            # initialize layer with random vals
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
            self.hidden_layers.append(hidden_layer)
        
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.output_layer.weight.data.uniform_(*hidden_init(self.output_layer))

        # move to GPU if available
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
        
    def _format(self, states):
        x = states
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, states):
        x = self._format(states)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x).squeeze()

    def np_value(self,states):
        value = self.forward(states).detach().cpu().numpy()
        return value
class PolicyPPO(nn.Module):
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
        super(PolicyPPO,self).__init__()
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
        self.output_layer_log_std.weight.data.uniform_(*hidden_init(self.output_layer_log_std))

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
        # x = self.batch_norm_layer(x)
        x_mean = self.output_layer_mean(x)
        # calculate the log of the standard deviation and clamp within reasonable values
        x_log_std = self.output_layer_log_std(x)
        x_log_std = torch.clamp(x_log_std, 
                                self.log_std_min, 
                                self.log_std_max)
        return x_mean, x_log_std
    
    def full(self, state, epsilon = 1e-6):
        """returns sampled action, log_prob of action, and greedy action for state"""
        # get mean and std log for states
        mean, log_std = self.forward(state)
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
        
        return action, log_prob, self.rescale_fn(torch.tanh(mean))

    def np_action(self, state, eps=0.5, noise=False):
        """returns an action and log probs in numpy format for environment step"""
        
        if np.random.random() < eps:
            mean, log_std = self.forward(state)
            policy = torch.distributions.Normal(mean, log_std.exp())
            action = self.rescale_fn(torch.tanh(mean))
            log_prob = policy.log_prob(mean)
            log_prob = log_prob.sum(-1).unsqueeze(-1)
            action_np = action.detach().cpu().numpy()
            log_prob_np = log_prob.detach().cpu().numpy()
        else:
            action, log_prob, mean = self.full(state)
            action_np = action.detach().cpu().numpy()
            log_prob_np = log_prob.detach().cpu().numpy()

        return action_np, log_prob_np

    def random_actions(self, state):
        mean, log_std = self.forward(state)
        policy = torch.distributions.Normal(mean, log_std.exp())
        action = torch.randn_like(mean).clamp(-0.9999,0.9999)
        pre_tanh_action = torch.atanh(action)
        log_prob = policy.log_prob(pre_tanh_action)
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        action_np = action.detach().cpu().numpy()
        log_prob_np = log_prob.detach().cpu().numpy()
        return action_np, log_prob_np

    def get_probs(self, state, action):
        """returns log probs and entropy for choosing provided action at given state"""
        mean, log_std = self.forward(state)
        # get a Normal distribution with those values
        policy = torch.distributions.Normal(mean, log_std.exp())
        # convert action back to pre-tanh value
        pre_tanh_action = torch.atanh(action)

        log_prob = policy.log_prob(pre_tanh_action)
        log_prob = log_prob.sum(-1).unsqueeze(-1)
        entropy = policy.entropy().unsqueeze(-1)
        return log_prob, entropy