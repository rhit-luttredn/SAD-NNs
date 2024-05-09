import torch
import torch.nn as nn

class SoftQNetwork(nn.Module):
    def __init__(self, 
                 observation_encoder:nn.Module, 
                 action_encoder:nn.Module,
                 memory_unit:nn.Module, 
                 q_network:nn.Module):
        super().__init__()
        self.observation_encoder = observation_encoder
        self.action_encoder = action_encoder
        self.memory_unit = memory_unit
        self.q_network = q_network
        
    def forward(self, x, a):
        x = self.observation_encoder(x)
        a = self.action_encoder(a)
        y = torch.cat([x, a], 1)
        # Here I need to unwrap for the LSTM
        y = self.memory_unit(y)
        q = self.q_network(y)
        return q

LOG_STD_MAX = 2
LOG_STD_MIN = -5
 
class Actor(nn.Module):
    def __init__(self, 
                 observation_encoder:nn.Module, 
                 memory_unit:nn.Module, 
                 network:nn.Module,
                 decoder_mean:nn.Module,
                 decoder_log_std:nn.Module,):
        super().__init__()
        self.observation_encoder = observation_encoder
        self.decoder_mean = decoder_mean
        self.decoder_log_std = decoder_log_std
        self.memory_unit = memory_unit
        self.network = network
    
    def forward(self, x):
        y = self.observation_encoder(x)
        # Here I need to unwrap for the LSTM
        y = self.memory_unit(y)
        y = self.network(y)
        mean = self.decoder_mean(y)
        log_std = torch.tanh(self.decoder_log_std(y))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        # t = (mean+log_std).cpu().detach().numpy()
        return mean, log_std
        
    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
        