import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import vector_to_parameters
import numpy as np
import random
from collections import deque
import os

## 格式還沒檢查
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        print("device:", self.device)

    # 等等檢查這邊格式，現在不做任何處理
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

# 用來 encode task seq 的資訊，這兩個block要檢查reward size 形式，其他都ok
class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1 + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    def forward(self, state, action, reward, next_state):
        # 這邊假設是reward大小為 [B,], 所以unsqueeze成 [B, 1], 但還要再檢查
        # decide whether to unsqueeze reward
        if reward.ndim == 1:
            reward = reward.unsqueeze(-1)
        x = torch.cat([state, action, reward, next_state], dim=-1)
        embedding = self.net(x)
        return embedding

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, recon_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, recon_dim),
        )
    def forward(self, embedding):
        return self.net(embedding)


## 這邊會輸出的是 actor and critic 的係數
class HyperNetwork(nn.Module):
    def __init__(self, latent_dim, hidden_dim, actor_param_size, critic_param_size):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.head_actor  = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, actor_param_size)
        )
        self.head_critic  = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, critic_param_size)
        )
    ## 這邊吃encoder task後產生的embedding
    def forward(self, embedding):
        hidden = self.trunk(embedding)
        actor  = self.head_actor(hidden)
        critic = self.head_critic(hidden)
        return actor, critic

# 5) Actor & Critic networks similarly parameterized
# Gaussian policy network (actor)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2, action_scale=1.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_scale = action_scale

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()  # reparameterization trick
        action = torch.tanh(x_t) * self.action_scale  # squashing function
        # Enforce action bounds
        action = torch.clamp(action, -self.action_scale, self.action_scale)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)

class Agent:
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim, action_scale, log_std_min, log_std_max, gamma, tau, alpha, device):
        self.device = device
        self.encoder = Encoder(state_dim, action_dim, hidden_dim, latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim, hidden_dim, state_dim + action_dim + 1 + state_dim).to(self.device)

        self.actor = Actor(state_dim, action_dim, hidden_dim, log_std_min, log_std_max, action_scale).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        print("actor params:", self.count_params(self.actor))
        print("critic params:", self.count_params(self.critic))
        self.hypernet = HyperNetwork(latent_dim,
                                     hidden_dim, 
                                     actor_param_size=self.count_params(self.actor), 
                                     critic_param_size=self.count_params(self.critic)).to(self.device)
        
        self.memory = deque(maxlen=10)
        self.replay_buffer = ReplayBuffer(1000000, device=device)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def count_params(self, module):
        return sum(p.numel() for p in module.parameters())

    def select_action(self, state, action, reward, next_state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state)
        self.memory.append((state, action, reward, next_state))
        states, actions, rewards, next_states = zip(*self.memory) 

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        embeddings = self.encoder(states, actions, rewards, next_states)
        # embeddings = embeddings.detach()
        embedding = torch.mean(embeddings, dim=0)
        actor_params, _ = self.hypernet(embedding)
        vector_to_parameters(actor_params, self.actor.parameters())
        
        if deterministic:
            mu, _ = self.actor(state)
            return mu.cpu().detach().numpy()
        else:
            action, _ = self.actor.sample(state)
            
            return action.cpu().detach().numpy()
    
    def update_actor_critic(self, replay_buffer, batch_size=256):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 1) Encoder + Decoder
        embedding = self.encoder(states, actions, rewards, next_states)
        recon = self.decoder(embedding)
        L_recon = F.mse_loss(recon, torch.cat([states, actions, rewards.unsqueeze(1), next_states], dim=-1))

        # 3) Hypernetwork
        actor_params, critic_params = self.hypernet(embedding)
        actor_params = torch.mean(actor_params, dim=0)
        critic_params = torch.mean(critic_params, dim=0)
        vector_to_parameters(actor_params, self.actor.parameters())
        vector_to_parameters(critic_params, self.critic.parameters())

        # 6) Compute SAC-style losses L_pi and L_Q on your imagined data…
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states) # Sample actions from policy for next_states
            q1_next, q2_next = self.critic(next_states, next_actions) ## [B , 1]
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs ## [B , 1]
            target_value = rewards + (1 - dones) * self.gamma * q_next.squeeze(-1).detach()
        
        # Update Q networks
        q1_pred, q2_pred = self.critic(states, actions)

        q1_pred = q1_pred.squeeze(-1)
        q2_pred = q2_pred.squeeze(-1) ## [Batchsize,]
        
        critic_loss = F.mse_loss(q1_pred, target_value) + F.mse_loss(q2_pred, target_value)

        # Update policy
        action_pi, log_pi = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, action_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_pi - q_pi).mean()

        # 7) Total & backward
        sac_loss = critic_loss + policy_loss
        
        return sac_loss, L_recon    
    

    def save(self, save_dir, episode):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.encoder.state_dict(), os.path.join(save_dir, f'encoder_{episode}.pth'))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir, f'decoder_{episode}.pth'))
        torch.save(self.hypernet.state_dict(), os.path.join(save_dir, f'hypernet_{episode}.pth'))
        # torch.save(self.actor.state_dict(), os.path.join(save_dir, f'actor_{episode}.pth'))
        # torch.save(self.critic.state_dict(), os.path.join(save_dir, f'critic_{episode}.pth'))
        # torch.save(self.critic_target.state_dict(), os.path.join(save_dir, f'critic_target_{episode}.pth'))
    
    def load(self, save_dir, episode):
        self.encoder.load_state_dict(torch.load(os.path.join(save_dir, f'encoder_{episode}.pth')))
        self.decoder.load_state_dict(torch.load(os.path.join(save_dir, f'decoder_{episode}.pth')))
        self.hypernet.load_state_dict(torch.load(os.path.join(save_dir, f'hypernet_{episode}.pth')))
        # self.actor.load_state_dict(torch.load(os.path.join(save_dir, f'actor_{episode}.pth')))
        # self.critic.load_state_dict(torch.load(os.path.join(save_dir, f'critic_{episode}.pth')))
        # self.critic_target.load_state_dict(torch.load(os.path.join(save_dir, f'critic_target_{episode}.pth')))
        self.encoder.eval()
        self.decoder.eval()
        self.hypernet.eval()
        # self.actor.eval()
        # self.critic.eval()
        # self.critic_target.eval()
