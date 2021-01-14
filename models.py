import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.state_fc = nn.Linear(state_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.state_fc1 = nn.Linear(512,256)
        self.layer_2 = nn.Linear(256, 128)
        #self.bn2 = nn.BatchNorm1d(128)
        self.layer_3 = nn.Linear(128, action_size)
        self.reset_parameters()
        return
    
    def reset_parameters(self):
        self.state_fc.weight.data.uniform_(*hidden_init(self.state_fc))
        self.state_fc1.weight.data.uniform_(*hidden_init(self.state_fc1))
        self.layer_2.weight.data.uniform_(*hidden_init(self.layer_2))
        self.layer_3.weight.data.uniform_(*hidden_init(self.layer_3))

    def forward(self, states):
        x = F.leaky_relu(self.bn1(self.state_fc(states)))
        x = F.leaky_relu(self.state_fc1(x))
        x = F.leaky_relu(self.layer_2(x))
        #x = F.leaky_relu(self.bn2(self.layer_2(x)))
        return torch.tanh(self.layer_3(x))

class Actor_SDPG(nn.Module):
    def __init__(self, state_size, action_size, dense1_size, dense2_size):
        super(Actor_SDPG,self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = nn.Linear(state_size, dense1_size).cuda()
        self.batchnorm = nn.BatchNorm1d(state_size).cuda()
        self.dense2 = nn.Linear(dense1_size, dense2_size).cuda()
        self.output_fc = nn.Linear(dense2_size,action_size).cuda()
        self.reset_parameters()
        return

    def reset_parameters(self):
        self.dense1.weight.data.uniform_(*hidden_init(self.dense1))
        self.dense2.weight.data.uniform_(*hidden_init(self.dense2))
        self.output_fc.weight.data.uniform_(-3e-3,3e-3)
        return
    def forward(self, states):
        x = self.batchnorm(states.view(-1,self.state_size))
        #x = F.leaky_relu(self.dense1(self.batchnorm(x)))
        x = F.leaky_relu(self.dense1(x))
        x = F.leaky_relu(self.dense2(x))
        x = torch.tanh(self.output_fc(x))
        # Scale tanh output to lower and upper action bounds
		# x = 0.5*((self.action_bound_high+self.action_bound_low) + x*(self.action_bound_high-self.action_bound_low))
        return x

class Critic_SDPG(nn.Module):
    def __init__(self, state_size, action_size, dense1_size, dense2_size, num_atoms, scope='critic'):
        super(Critic_SDPG, self).__init__()
        # self.v_min = v_min
        # self.v_max = v_max
        self.scope = scope
        self.state_size = state_size
        self.action_size = action_size
        self.dense1_size = dense1_size
        self.dense2_size = dense2_size
        self.num_atoms = num_atoms
        #self.phi_embedding_size = phi_embedding_size
        self.dense1 = nn.Linear(state_size, dense1_size).cuda()
        self.batchnorm = nn.BatchNorm1d(state_size).cuda()
        #self.dense2a = nn.Linear(dense1_size+action_size, dense2_size).cuda()
        self.dense2a = nn.Linear(dense1_size, dense2_size).cuda()
        self.dense2b = nn.Linear(action_size, dense2_size).cuda()
        self.dense2c = nn.Linear(1, dense2_size).cuda()
        self.dense3 = nn.Linear(dense2_size, dense2_size).cuda()
        self.output_fc = nn.Linear(dense2_size,1).cuda()
        self.reset_parameters()
        return

    def reset_parameters(self):
        self.dense1.weight.data.uniform_(*hidden_init(self.dense1))
        self.dense2a.weight.data.uniform_(*hidden_init(self.dense2a))
        self.dense2b.weight.data.uniform_(*hidden_init(self.dense2b))
        self.dense2c.weight.data.uniform_(*hidden_init(self.dense2c))
        self.dense3.weight.data.uniform_(*hidden_init(self.dense3))
        self.output_fc.weight.data.uniform_(-3e-3,3e-3)
        return
    
    
    def forward(self, states, actions, samples ):
        xs = states.view(-1,self.state_size)
        xs = F.leaky_relu(self.dense1(self.batchnorm(xs)))
        xs = self.dense2a(xs)
        xs.unsqueeze_(1)
        xa = self.dense2b(actions)
        xa.unsqueeze_(1)
        x = torch.add(xs,xa)
        x_s = self.dense2c(samples).view(-1,self.num_atoms,self.dense2_size)
        x = torch.add(x, x_s)
        x = F.leaky_relu(self.dense3(x)).view(-1,self.dense2_size)
        # self.dense2 = tf.reshape(self.dense2, [batch_size*num_atoms, dense2_size])
        
        x = self.output_fc(x)
        # scale the output samples 
        # self.output_samples = 0.5 * (x*(self.v_max - self.v_min)+ (self.v_max + self.v_min))
        return x.view(-1,self.num_atoms)
    """
    def forward(self, states, actions, samples):
        xs = self.batchnorm(states.view(-1,self.state_size))
        #xs = F.leaky_relu(self.dense1(self.batchnorm(xs)))
        xs = F.leaky_relu(self.dense1(xs))
        x = torch.cat((xs, actions), dim=1)
        x = F.leaky_relu(self.dense2a(x)).unsqueeze(1)
        xa = self.dense2c(samples).view(-1,self.num_atoms,self.dense2_size)
        xb = x+xa
        x = F.leaky_relu(self.dense3(xb)).view(-1,self.dense2_size)
        x = self.output_fc(x)
        #x = F.leaky_relu(self.value_fc2(x))
        return x.view(-1,self.num_atoms)
    """
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.state_fc = nn.Linear(state_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.state_fc1 = nn.Linear(512,256)
        self.value_fc1 = nn.Linear(256 +action_size, 256)
        self.value_fc2 = nn.Linear(256,128)
        #self.bn2 = nn.BatchNorm1d(128)
        self.output_fc = nn.Linear(128,1)
        self.reset_parameters()
        return
    
    def reset_parameters(self):
        self.state_fc.weight.data.uniform_(*hidden_init(self.state_fc))
        self.state_fc1.weight.data.uniform_(*hidden_init(self.state_fc1))
        self.value_fc1.weight.data.uniform_(*hidden_init(self.value_fc1))
        self.value_fc2.weight.data.uniform_(*hidden_init(self.value_fc2))
        
        self.output_fc.weight.data.uniform_(*hidden_init(self.output_fc))

    def forward(self, states, action):
        xs = F.leaky_relu(self.bn1(self.state_fc(states)))
        xs = F.leaky_relu(self.state_fc1(xs))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.value_fc1(x))
        x = F.leaky_relu(self.value_fc2(x))
        #x = F.leaky_relu(self.bn2(self.value_fc2(x)))
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