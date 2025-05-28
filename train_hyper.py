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
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()

        self.actor_fc1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1152),
        )
        self.actor_fc2 = nn.Sequential(
            nn.Linear(latent_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 4160),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(latent_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 390),
        )
        self.actor_log_std = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 390),
        )

        self.critic_q11 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1536), 
        )
        self.critic_q12  = nn.Sequential(
            nn.Linear(latent_dim, 2*hidden_dim), 
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 4160), 
        )
        self.critic_q13  = nn.Sequential(
            nn.Linear(latent_dim, 16), 
            nn.ReLU(),
            nn.Linear(16, 65), 
        )

        self.critic_q21 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1536), 
        )
        self.critic_q22  = nn.Sequential(
            nn.Linear(latent_dim, 2*hidden_dim), 
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 4160), 
        )
        self.critic_q23  = nn.Sequential(
            nn.Linear(latent_dim, 16), 
            nn.ReLU(),
            nn.Linear(16, 65), 
        )

    ## 這邊吃encoder task後產生的embedding
    def forward(self, embedding):
        actor_fc1  = self.actor_fc1(embedding)
        actor_fc2  = self.actor_fc2(embedding)
        actor_mean  = self.actor_mean(embedding)
        actor_log_std = self.actor_log_std(embedding)

        critic_q11 = self.critic_q11(embedding)
        critic_q12 = self.critic_q12(embedding)
        critic_q13 = self.critic_q13(embedding)

        critic_q21 = self.critic_q21(embedding)
        critic_q22 = self.critic_q22(embedding)
        critic_q23 = self.critic_q23(embedding)
        return actor_fc1, actor_fc2, actor_mean, actor_log_std, critic_q11, critic_q12, critic_q13, critic_q21, critic_q22, critic_q23
    
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
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)

        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2., dtype=torch.float32).to(device)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2., dtype=torch.float32).to(device)

        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample() 
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
      
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.apply(weights_init_)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1).to(torch.float32)
        return self.q1(sa), self.q2(sa)

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
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
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
        self.hidden_dim = 4096
        self.latent_dim = 1024
        self.encoder = Encoder(state_dim, action_dim, self.hidden_dim, self.latent_dim).to(self.device)
        self.decoder = Decoder(self.latent_dim, self.hidden_dim , state_dim + action_dim + 1 + state_dim).to(self.device)
        print("actor params:", self.count_params(self.actor))
        print("critic params:", self.count_params(self.critic))
        self.hypernet = HyperNetwork(self.latent_dim, self.hidden_dim).to(self.device)
        
        self.optimizer_encoder = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=3e-4)
        self.optimizer_hyper = optim.Adam(self.hypernet.parameters(), lr=8e-4)

    def count_params(self, module):
        return sum(p.numel() for p in module.parameters())
        
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)

        # 抽幾組出來算embedding
        state, action, reward, next_state, done = self.memory.sample(30)
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        state, action, reward, next_state, done = [x.to(self.device) for x in (state, action, reward, next_state, done)]
        with torch.no_grad():
            embeddings = self.encoder(state, action, reward, next_state)
            embedding = torch.mean(embeddings, dim=0)
            # 重建actor係數
            actor_fc1, actor_fc2, actor_mean, actor_log_std, _, _, _, _, _, _ = self.hypernet(embedding)
            vector_to_parameters(actor_fc1, self.actor.fc1.parameters())
            vector_to_parameters(actor_fc2, self.actor.fc2.parameters())
            vector_to_parameters(actor_mean, self.actor.mean.parameters())
            vector_to_parameters(actor_log_std, self.actor.log_std.parameters())

            if deterministic:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def train(self):
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = [x.to(self.device) for x in (state, action, reward, next_state, done)]
        embeddings = self.encoder(state, action, reward, next_state)
        recon = self.decoder(embeddings)
        L_recon = F.mse_loss(recon, torch.cat([state, action, reward.unsqueeze(1), next_state], dim=-1))

        avg_embedding = torch.mean(embeddings, dim=0, keepdim=True)  # shape: [1, D]
         
        actor_fc1, actor_fc2, actor_mean, actor_log_std, critic_q11, critic_q12, critic_q13, critic_q21, critic_q22, critic_q23 = self.hypernet(avg_embedding) # [B, 1]一組填入係數即可
        actor_fc1     = actor_fc1.squeeze(0)
        actor_fc2     = actor_fc2.squeeze(0)
        actor_mean    = actor_mean.squeeze(0)
        actor_log_std = actor_log_std.squeeze(0)

        critic_q11 = critic_q11.squeeze(0)
        critic_q12 = critic_q12.squeeze(0)
        critic_q13 = critic_q13.squeeze(0)

        critic_q21 = critic_q21.squeeze(0)
        critic_q22 = critic_q22.squeeze(0)
        critic_q23 = critic_q23.squeeze(0)

        
        vector_to_parameters(actor_fc1,     list(self.actor.fc1.parameters()))
        vector_to_parameters(actor_fc2,     list(self.actor.fc2.parameters()))
        vector_to_parameters(actor_mean,    list(self.actor.mean.parameters()))
        vector_to_parameters(actor_log_std, list(self.actor.log_std.parameters()))

        vector_to_parameters(critic_q11, list(self.critic.q1[0].parameters())) 
        vector_to_parameters(critic_q12, list(self.critic.q1[2].parameters()))  
        vector_to_parameters(critic_q13, list(self.critic.q1[4].parameters()))  

        vector_to_parameters(critic_q21, list(self.critic.q2[0].parameters()))  
        vector_to_parameters(critic_q22, list(self.critic.q2[2].parameters()))  
        vector_to_parameters(critic_q23, list(self.critic.q2[4].parameters()))

        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            y = reward + self.gamma * (1 - done) * target_q.squeeze(-1)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1.squeeze(-1), y) + F.mse_loss(q2.squeeze(-1), y)

        self.optimizer_encoder.zero_grad()
        L_recon.backward()
        self.optimizer_encoder.step()

        new_action, log_prob, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()
        
        
        self.optimizer_hyper.zero_grad()
        critic_loss.backward(retain_graph=True) 
        for name, param in self.hypernet.named_parameters():
            if "critic" in name:
                param.grad = param.grad  # 保留 critic 部分梯度
            else:
                param.grad = None        # 清空其他 block 梯度
        self.optimizer_hyper.step()

        self.optimizer_hyper.zero_grad()
        actor_loss.backward()
        for name, param in self.hypernet.named_parameters():
            if "actor" in name:
                param.grad = param.grad
            else:
                param.grad = None
        self.optimizer_hyper.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.train_step +=1
        if self.train_step % self.update_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
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
