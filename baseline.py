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
from halfcheetah_vel_env import HalfCheetahVelEnv


class JointFailureWrapper(Wrapper):
    def __init__(self, env, failed_joint):
        if not hasattr(env, 'reward_range'):
            env.reward_range = (-np.inf, np.inf)
        super().__init__(env)
        self.failed_joint = failed_joint

    def step(self, action):
        a = np.array(action, copy=True)
        a[self.failed_joint] = 0.0
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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.log_alpha = torch.tensor(-1.0, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)

        self.target_entropy = torch.tensor(-action_dim, dtype=torch.float32).to(self.device)
        self.gamma = 0.99
        self.tau = 0.005
        self.memory = ReplayBuffer(1000000)
        self.batch_size = 256
        self.train_step = 0
        self.update_freq = 1

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def train(self):
        # if len(self.memory.buffer) < self.batch_size:
        #     return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = [x.to(self.device) for x in (state, action, reward, next_state, done)]

        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            y = reward + self.gamma * (1 - done) * target_q.squeeze(-1)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1.squeeze(-1), y) + F.mse_loss(q2.squeeze(-1), y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

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
    failed_joints     = [1, 3, 5]
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
  
    # agent = SACAgent(state_dim, action_dim, env.action_space)
    # agent.load(load_path, train=True)
    num_episodes = 200
    reward_history = [] 
    warmup_episode = 50
    trained_tasks = []

    for name, env in env_list:
        print(f"Training on failure joint = {name}")
        trained_tasks.append((name, env))
        agent = SACAgent(state_dim, action_dim, env.action_space)
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
                agent.save(f"checkpoints/sac_{episode+1}_{name}")
                avg_reward = np.mean(reward_history[-20:])
                print(f'"Episode {episode + 1}/{num_episodes}, Total reward: {total_reward:.2f}, joint: {name}')

                if (episode + 1) % 100 == 0:
                    print("=== Evaluation across all tasks ===")
                    for test_name, test_env in trained_tasks:
                        mean_r, std_r = evaluate_agent(agent, test_env, episodes=5)
                        print(f"Velocity: [{test_name}] avg reward: {mean_r:.2f} ± {std_r:.2f}")
