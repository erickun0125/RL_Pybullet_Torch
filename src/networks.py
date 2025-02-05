import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 256):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std
    
    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.sum(1, keepdim=True)
        log_prob -= torch.log(1 - y_t.pow(2) + epsilon)

        return action, log_prob,mean
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 256):
        super(Critic, self).__init__()
        #Q1 architecture
        self.linear1_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear1_3 = nn.Linear(hidden_dim, 1)

        #Q2 architecture
        self.linear2_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_3 = nn.Linear(hidden_dim, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.linear1_1(sa))
        q1 = F.relu(self.linear1_2(q1))
        q1 = self.linear1_3(q1)

        q2 = F.relu(self.linear2_1(sa))
        q2 = F.relu(self.linear2_2(q2))
        q2 = self.linear2_3(q2)

        return q1, q2