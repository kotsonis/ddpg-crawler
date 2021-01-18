import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging
from absl import flags
config = flags.FLAGS
flags.DEFINE_float(
                  name='dual_constraint',
                  default=0.1,
                  help='hard constraint of the E-step')

flags.DEFINE_integer(
                name='actor_dim_dense_1',
                default=512,
                help='output dimension of actor 1st hidden layer')
flags.DEFINE_integer(
                name='actor_dim_dense_2',
                default=256,
                help='output dimension of actor 2nd hidden layer')
flags.DEFINE_integer(
                name='actor_dim_dense_3',
                default=128,
                help='output dimension of actor 3nd hidden layer')
flags.DEFINE_integer(
                name='critic_dim_dense_1',
                default=512,
                help='output dimension of critic 1st hidden layer')
flags.DEFINE_integer(
                name='critic_dim_dense_2',
                default=256,
                help='output dimension of critic 2nd hidden layer')
flags.DEFINE_integer(
                name='critic_dim_dense_3',
                default=128,
                help='output dimension of critic 3nd hidden layer')


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
            actions = self.forward(states)
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


"""
class IQN(nn.Module):
    def __init__(self, state_size, action_size, N, dueling=False, device="cuda:0"):
        super(IQN, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.N = N  
        self.n_cos = 64 # based on recommendation from Dabney et all, 2018, arXiv: 1806.06923v1

        self.layer_size = 256
        # create the PI * i vector to be used in eq. (4) of arXiv: 1806.06923v1 (phi(tau) = relu(Sum_i(cos(PI*i*tau)w_ij+b_j)))
        self.pis = torch.FloatTensor([np.pi*i for i in range(0,self.n_cos)])
        self.pis = self.pis.view(1,1,self.n_cos) #reshape
        
        # Network Architecture
        self.state = nn.Linear(self.input_shape,self.layer_size)
        self.head = nn.Linear(self.action_size+self.layer_size, self.layer_size) 
        self.phi_tau_layer = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, 1)    
        
        # initialize weights to random values
        #weight_init([self.head_1, self.ff_1])

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]
        
    def calc_cos(self, batch_size, n_tau=32):
       
        Calculating the cosinus values depending on the number of tau samples
       
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device) #(batch_size, n_tau, 1)  .to(self.device)
        cos = torch.cos(taus*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus
    
    def forward(self, input, action, num_tau=8):
        
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        
        
        batch_size = input.shape[0]
        x = torch.relu(self.state(input))
        x = torch.cat((x,action), dim=1)
        x = torch.relu(self.head(x))
        # sample number of tau
        taus = torch.rand(batch_size,num_tau).unsqueeze(-1) # shape is (batch_size,num_tau,1)
        cos = torch.cos(taus*self.pis) # shape is (batch_size,num_tau,num_cos)
        cos = cos.view(batch_size*num_tau, self.n_cos) # shape is now (batch_size*num_tau, num_cos)
        
        cos_x = torch.relu(self.phi_tau_layer(cos)).view(batch_size, num_tau, self.layer_size) # shape (batch_size, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer) and multiply with cos_x embedding

        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.layer_size)  #batch_size*num_tau, self.layer_size
        
        x = torch.relu(self.ff_1(x))    # calculate first combined hidden layer

        out = self.ff_2(x)
        
        return out.view(batch_size, num_tau, 1), taus
    
    def get_qvalues(self, inputs, action):
        quantiles, _ = self.forward(inputs, action, self.N)
        actions = quantiles.mean(dim=1)
        return actions  
"""
"""
class BaseModel(nn.Module):
    
    def __init__(self):
        super().__init__()

    def sample_noise(self):
        if self.noisy_net:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.sample()

class IQN(BaseModel):    
    def __init__(self, num_channels, num_actions, K=32, num_cosines=32,
                 embedding_dim=7*7*64, dueling_net=False, noisy_net=False):
        super(IQN, self).__init__()

        # Feature extractor of DQN.
        self.dqn_net = DQNBase(num_channels=num_channels)
        # Cosine embedding network.
        self.cosine_net = CosineEmbeddingNetwork(
            num_cosines=num_cosines, embedding_dim=embedding_dim,
            noisy_net=noisy_net)
        # Quantile network.
        self.quantile_net = QuantileNetwork(
            num_actions=num_actions, dueling_net=dueling_net,
            noisy_net=noisy_net)

        self.K = K
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

    def calculate_state_embeddings(self, states):
        return self.dqn_net(states)

    def calculate_quantiles(self, taus, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(state_embeddings, tau_embeddings)

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        # Sample fractions.
        taus = torch.rand(
            batch_size, self.K, dtype=state_embeddings.dtype,
            device=state_embeddings.device)

        # Calculate quantiles.
        quantiles = self.calculate_quantiles(
            taus, state_embeddings=state_embeddings)
        assert quantiles.shape == (batch_size, self.K, self.num_actions)

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q
"""