import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.distributions import Normal
import os
import pickle
import sys
from gym import Wrapper
from torch.nn.utils import vector_to_parameters
import ipdb
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
    
class HyperNetwork(nn.Module):
    def __init__(self, latent_dim, trunk_dim=1024):
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, trunk_dim),
            nn.ReLU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU(),
        )

        # Actor head outputs
        self.actor_fc1_head     = nn.Linear(trunk_dim, 1152)  # 64 * state_dim + 64 (bias)
        self.actor_fc2_head     = nn.Linear(trunk_dim, 4160)  # 64 * 64 + 64 (bias)
        self.actor_mean_head    = nn.Linear(trunk_dim, 390)   # action_dim * 64 + action_dim
        self.actor_logstd_head  = nn.Linear(trunk_dim, 390)

        # Critic Q1 heads
        self.critic_q1_fc1_head = nn.Linear(trunk_dim, 1536)  # 64 * (state+action) + 64
        self.critic_q1_fc2_head = nn.Linear(trunk_dim, 4160)
        self.critic_q1_fc3_head = nn.Linear(trunk_dim, 65)    # 1 * 64 + 1

        # Critic Q2 heads
        self.critic_q2_fc1_head = nn.Linear(trunk_dim, 1536)
        self.critic_q2_fc2_head = nn.Linear(trunk_dim, 4160)
        self.critic_q2_fc3_head = nn.Linear(trunk_dim, 65)

    def forward(self, embedding):
        h = self.trunk(embedding)

        actor_fc1     = self.actor_fc1_head(h)
        actor_fc2     = self.actor_fc2_head(h)
        actor_mean    = self.actor_mean_head(h)
        actor_log_std = self.actor_logstd_head(h)

        critic_q1_fc1 = self.critic_q1_fc1_head(h)
        critic_q1_fc2 = self.critic_q1_fc2_head(h)
        critic_q1_fc3 = self.critic_q1_fc3_head(h)

        critic_q2_fc1 = self.critic_q2_fc1_head(h)
        critic_q2_fc2 = self.critic_q2_fc2_head(h)
        critic_q2_fc3 = self.critic_q2_fc3_head(h)

        return actor_fc1, actor_fc2, actor_mean, actor_log_std, critic_q1_fc1, critic_q1_fc2, critic_q1_fc3, critic_q2_fc1, critic_q2_fc2, critic_q2_fc3
    
class JointFailureWrapper(Wrapper):
    def __init__(self, env, failed_joint):
        if not hasattr(env, 'reward_range'):
            env.reward_range = (-np.inf, np.inf)
        super().__init__(env)
        self.failed_joint = failed_joint

    def step(self, action):
        a = np.array(action, copy=True)
        a[self.failed_joint[0]] = 0.0
        a[self.failed_joint[1]] = 0.0
        return self.env.step(a)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_space=None, device='cuda'):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2., dtype=torch.float32).to(device)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2., dtype=torch.float32).to(device)

        self.apply(weights_init_)

    def forward(self, state, actor_fc1, actor_fc2, actor_mean, actor_log_std):
        w1 = actor_fc1[:self.state_dim * 64].view(64, self.state_dim)
        b1 = actor_fc1[self.state_dim * 64:]
        x = F.linear(state, w1, b1)
        x = F.relu(x)

        w2 = actor_fc2[:64 * 64].view(64, 64)
        b2 = actor_fc2[64 * 64:]
        x = F.linear(x, w2, b2)
        x = F.relu(x)

        w_m = actor_mean[:self.action_dim * 64].view(self.action_dim, 64)
        b_m = actor_mean[self.action_dim * 64:]
        mean = F.linear(x, w_m, b_m)

        w_s = actor_log_std[:self.action_dim * 64].view(self.action_dim, 64)
        b_s = actor_log_std[self.action_dim * 64:]
        log_std = F.linear(x, w_s, b_s).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state, actor_fc1, actor_fc2, actor_mean, actor_log_std):
        mean, std = self.forward(state, actor_fc1, actor_fc2, actor_mean, actor_log_std)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
      
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.in_dim = state_dim + action_dim

    def q_forward(self, sa, critic_fc1, critic_fc2, critic_fc3):
        w1 = critic_fc1[:self.in_dim * 64].view(64, self.in_dim)
        b1 = critic_fc1[self.in_dim * 64:]
        x = F.linear(sa, w1, b1)
        x = F.relu(x)

        w2 = critic_fc2[:64 * 64].view(64, 64)
        b2 = critic_fc2[64 * 64:]
        x = F.linear(x, w2, b2)
        x = F.relu(x)

        w3 = critic_fc3[:64 * 1].view(1, 64)
        b3 = critic_fc3[64 * 1:]
        q = F.linear(x, w3, b3)
        return q
    
    def forward(self, sa, q1_fc1, q1_fc2, q1_fc3, q2_fc1, q2_fc2, q2_fc3):
            q1 = self.q_forward(sa, q1_fc1, q1_fc2, q1_fc3)
            q2 = self.q_forward(sa, q2_fc1, q2_fc2, q2_fc3)
            return q1, q2

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.int32)
       
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)

        return states, actions, rewards, next_states, dones
    
    def clear(self):
        self.buffer.clear()

class SACAgent:
    def __init__(self, state_dim, action_dim, action_space):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, action_space, self.device).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        self.log_alpha = torch.tensor(-1.0, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)

        self.target_entropy = torch.tensor(-action_dim, dtype=torch.float32).to(self.device)
        self.gamma = 0.99
        self.tau = 0.005
        self.memory = ReplayBuffer(1000000)
        self.batch_size = 256
        self.train_step = 0
        self.update_freq = 1

        # hypernet part
        self.hidden_dim = 1024
        self.latent_dim = 1024
        self.encoder = Encoder(state_dim, action_dim, self.hidden_dim, self.latent_dim).to(self.device)
        self.decoder = Decoder(self.latent_dim, self.hidden_dim , state_dim + action_dim + 1 + state_dim).to(self.device)
        self.hypernet = HyperNetwork(self.latent_dim, self.hidden_dim).to(self.device)
        self.hypernet_target = HyperNetwork(self.latent_dim, self.hidden_dim).to(self.device)
        self.hypernet_target.load_state_dict(self.hypernet.state_dict())
        
        self.optimizer_encoder = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=3e-4)
        actor_params = []
        critic_params = []
        for name, p in self.hypernet.named_parameters():
            if name.startswith("trunk"):
                actor_params.append(p)
                critic_params.append(p)
            elif name.startswith("actor_"):
                actor_params.append(p)
            else:
                critic_params.append(p)

        # 两个 Optimizer：一个只更新 actor_params，一个只更新 critic_params
        self.optimizer_hyper_actor  = torch.optim.Adam(actor_params,  lr=8e-4)
        self.optimizer_hyper_critic = torch.optim.Adam(critic_params, lr=8e-4)

    def count_params(self, module):
        return sum(p.numel() for p in module.parameters())
        
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        obs_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)

        # 抽幾組出來算embedding
        state, action, reward, next_state, done = self.memory.sample(30)
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        state, action, reward, next_state, done = [x.to(self.device) for x in (state, action, reward, next_state, done)]
        with torch.no_grad():
            embeddings = self.encoder(state, action, reward, next_state)
            embedding = torch.mean(embeddings, dim=0)
            # 重建actor係數
            out = self.hypernet(embedding)
            actor_fc1, actor_fc2, actor_mean, actor_log_std, *_ = [o.squeeze(0) for o in out]

            if deterministic:
                _, _, action = self.actor.sample(obs_tensor, actor_fc1, actor_fc2, actor_mean, actor_log_std)
            else:
                action, _, _ = self.actor.sample(obs_tensor, actor_fc1, actor_fc2, actor_mean, actor_log_std)
        return action.detach().cpu().numpy()[0]
    
    def train(self):
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = [x.to(self.device) for x in (state, action, reward, next_state, done)]

        embeddings = self.encoder(state, action, reward, next_state)
        recon = self.decoder(embeddings)
        L_recon = F.mse_loss(recon, torch.cat([state, action, reward.unsqueeze(1), next_state], dim=-1))

        self.optimizer_encoder.zero_grad()
        L_recon.backward()
        self.optimizer_encoder.step()

        avg_embedding = torch.mean(embeddings, dim=0, keepdim=True).detach()  # shape: [1, D], ##這邊先更新，因為擔心會被後面的graph影響到
        output = self.hypernet(avg_embedding) # [B, 1]一組填入係數即可
        actor_fc1, actor_fc2, actor_mean, actor_log_std, critic_q11, critic_q12, critic_q13, critic_q21, critic_q22, critic_q23 = [o.squeeze(0) for o in output]

        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_state, actor_fc1, actor_fc2, actor_mean, actor_log_std)
            out_target = self.hypernet_target(avg_embedding)
            _, _, _, _, target_q11, target_q12, target_q13, target_q21, target_q22, target_q23 = [o.squeeze(0) for o in out_target]
            nsa = torch.cat([next_state, next_action], dim = -1)
            target_q1, target_q2 = self.critic(nsa, target_q11, target_q12, target_q13, target_q21, target_q22, target_q23)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            y = reward + self.gamma * (1 - done) * target_q.squeeze(-1)


        sa = torch.cat([state, action], dim = -1)
        q1, q2 = self.critic(sa, critic_q11, critic_q12, critic_q13, critic_q21, critic_q22, critic_q23)
        critic_loss = F.mse_loss(q1.squeeze(-1), y) + F.mse_loss(q2.squeeze(-1), y)

        self.optimizer_hyper_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_hyper_critic.step()


        new_output = self.hypernet(avg_embedding)
        n_actor_fc1, n_actor_fc2, n_actor_mean, n_actor_log_std, n_critic_q11, n_critic_q12, n_critic_q13, n_critic_q21, n_critic_q22, n_critic_q23 = [o2.squeeze(0) for o2 in new_output]
        new_action, log_prob, _ = self.actor.sample(state, n_actor_fc1, n_actor_fc2, n_actor_mean, n_actor_log_std)
        sna = torch.cat([state, new_action], dim = -1)

        with torch.no_grad():
            q1_pi, q2_pi = self.critic(sna, n_critic_q11, n_critic_q12, n_critic_q13, n_critic_q21, n_critic_q22, n_critic_q23)
        actor_loss = (self.alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()
        
        # —— 更新 Actor Head —— 
        self.optimizer_hyper_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_hyper_actor.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.train_step +=1
        if self.train_step % self.update_freq == 0:
            for param, target_param in zip(self.hypernet.parameters(), self.hypernet_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path_prefix):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        torch.save({'actor': self.actor.state_dict(),}, f"{path_prefix}_model.pth")

    def load(self, path_prefix):
        checkpoint = torch.load(f"{path_prefix}_model.pth", map_location=torch.device('cpu'))
        self.actor.load_state_dict(checkpoint['actor'])

def evaluate_agent(agent, env, episodes=5):

    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            # deterministic / eval mode
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)
     
if __name__ == "__main__":
    
    target_velocities = [0.5, 1.0, 1.5]
    # 要失效的關節索引（HalfCheetah-v2 一共有 6 個 actuator，你可以依序指定 0~5）
    failed_joints     = [(1, 3) , (2, 5), (0, 4)]
    env_list = []
    env = gym.make('HalfCheetah-v4')
    env_list.append((f'HalfCheetah_joint_normal', env))

    for joint in failed_joints:

        env = gym.make('HalfCheetah-v4')

        env = JointFailureWrapper(env, failed_joint=joint)

        name = f'HalfCheetah_joint{joint}'
        env_list.append((name, env))
    # base_env = HalfCheetahVelEnv()
    # tasks    = base_env.sample_tasks(num_tasks=5)    # 比如取 5 種速度

    # 2) 為每個 task 建立一個 env 實例
    # envs = []
    # for task in tasks:
    #     env = HalfCheetahVelEnv(task)
    #     env.reset()      # 會設置 _goal_vel
    #     envs.append((task['velocity'], env))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
  
    agent = SACAgent(state_dim, action_dim, env.action_space)
    # agent.load(load_path, train=True)
    num_episodes = 200
    reward_history = [] 
    warmup_episode = 50
    trained_tasks = []

    for name, env in env_list:
        print(f"Training on failure joint = {name}")
        trained_tasks.append((name, env))
        agent.memory.clear()
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False

            # for _ in range(max_step):
            while not done:
                if episode <= warmup_episode:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.memory.add(state, action, reward, next_state, done)

                if episode >  warmup_episode:
                    agent.train()
                    
                state = next_state
                total_reward += reward
                # total_step += 1

            # print(f"Episode {episode + 1} Reward: {total_reward:.2f}")
            reward_history.append(total_reward)

            if (episode + 1) % 20 == 0:
                # torch.save(agent.model.state_dict(), f"checkpoints/sac_{episode+1}.pth")
                agent.save(f"checkpoints/sac_{episode+1}")
                avg_reward = np.mean(reward_history[-20:])
                print(f'"Episode {episode + 1}/{num_episodes}, Total reward: {total_reward:.2f}, joint: {name}')

                if (episode + 1) % 100 == 0:
                    print("=== Evaluation across all tasks ===")
                    for test_name, test_env in trained_tasks:
                        mean_r, std_r = evaluate_agent(agent, test_env, episodes=5)
                        print(f"Velocity: [{test_name}] avg reward: {mean_r:.2f} ± {std_r:.2f}")
