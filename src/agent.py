# src/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from networks import Actor, Critic
from replay_buffer import ReplayBuffer

class SACAgent:
    def __init__(self, state_dim, action_dim, device, 
                 gamma=0.99, tau=0.005, 
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 buffer_max_size=1000000, batch_size=256):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim

        # Actor 네트워크와 최적화기
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic 네트워크 및 타깃 네트워크, 최적화기
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 엔트로피 온도 파라미터 (log_alpha)와 자동 튜닝 옵티마이저
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().item()
        self.target_entropy = -action_dim  # 목표 엔트로피

        # 경험 리플레이 버퍼
        self.replay_buffer = ReplayBuffer(max_size=buffer_max_size)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor.forward(state)
                action = torch.tanh(mean)
                return action.cpu().numpy()[0]
        else:
            with torch.no_grad():
                action, _, _ = self.actor.sample(state)
                return action.cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer.storage) < self.batch_size:
            return

        # 배치 샘플링
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Critic 업데이트
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 업데이트
        action_new, log_prob_new, _ = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.log_alpha.exp() * log_prob_new - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 자동 엔트로피 튜닝 업데이트
        alpha_loss = -(self.log_alpha * (log_prob_new + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # 타깃 네트워크 소프트 업데이트
        self.soft_update(self.critic_target, self.critic, self.tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
