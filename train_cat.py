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
import ipdb

# class Encoder(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim + action_dim + 1 + state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, latent_dim),
#         )
#     def forward(self, state, action, reward, next_state):
#         # 這邊假設是reward大小為 [B,], 所以unsqueeze成 [B, 1], 但還要再檢查
#         # decide whether to unsqueeze reward
#         if reward.ndim == 1:
#             reward = reward.unsqueeze(-1)
#         x = torch.cat([state, action, reward, next_state], dim=-1)
#         embedding = self.net(x)
#         return embedding
    
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_tasks):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.aggregator = nn.GRU(hidden_dim, embedding_dim, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_tasks)  # output logits
        )

    def forward(self, context_batch):
        """
        context_batch: [B, K, input_dim] → each is [s, a, r, s']
        return:
            task_probs: [B, num_tasks] (softmax 分布，可給 actor 用)
            logits: [B, num_tasks]（可選：給 cross-entropy loss 用）
            embedding: [B, embedding_dim]（可選：debug 或可視化用）
        """
        B, K, D = context_batch.shape
        feat = self.trunk(context_batch)  # [B, K, hidden_dim]
        _, h = self.aggregator(feat)      # h: [1, B, embedding_dim]
        z = h.squeeze(0)                  # [B, embedding_dim]

        logits = self.classifier(z)       # [B, num_tasks]
        probs = F.softmax(logits, dim=-1) # [B, num_tasks]
        return probs, logits, z
    
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
    def __init__(self, state_dim, action_dim, task_embed_dim ,action_space=None, device='cuda'):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim + task_embed_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)

        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2., dtype=torch.float32).to(device)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2., dtype=torch.float32).to(device)

        self.apply(weights_init_)

    def forward(self, state, task_info):
        x = torch.cat([state, task_info], dim = -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state, task_info):
        mean, std = self.forward(state, task_info)
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
    def __init__(self, state_dim, action_dim, task_embed_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim + task_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim + task_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.apply(weights_init_)

    def forward(self, state, action, task_embed_dim):
        sa = torch.cat([state, action, task_embed_dim], dim=-1).to(torch.float32)
        return self.q1(sa), self.q2(sa)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32) ## [1, ]
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.int32)
       
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards) # [B, ]
        next_states = torch.stack(next_states)
        dones = torch.stack(dones) # [B, ]

        return states, actions, rewards, next_states, dones
    
    def clear(self):
        self.buffer.clear()

class SACAgent:
    def __init__(self, state_dim, action_dim, embedding_dim, action_space):
        self.num_tasks = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, self.num_tasks, action_space, self.device).to(self.device)
        self.critic = Critic(state_dim, action_dim, self.num_tasks).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, self.num_tasks).to(self.device)
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

        self.hidden_dim = 512
        context_dim = state_dim + action_dim + 1 + state_dim
        self.encoder = Encoder(context_dim, self.hidden_dim, embedding_dim, self.num_tasks).to(self.device)
        self.decoder = Decoder(embedding_dim, self.hidden_dim, context_dim).to(self.device)
        self.optimizer_encoder = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=3e-4)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        obs_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        # 這邊沒有把done算進去，我覺得可以一起算，但先不弄，晚點再說
        state, action, reward, next_state, _ = self.memory.sample(100)
        reward = reward.unsqueeze(-1)
        context = torch.cat([state, action, reward, next_state], dim=-1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            task_probs, _, _ = self.encoder(context)
            task_info = F.one_hot(task_probs.argmax(dim=-1), num_classes = self.num_tasks).float()
            if deterministic:
                _, _, action = self.actor.sample(obs_tensor, task_info)
            else:
                action, _, _ = self.actor.sample(obs_tensor, task_info)
        return action.detach().cpu().numpy()[0]

    def train(self, task_id):
    # === 取出 batch from buffer ===
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = [x.to(self.device) for x in (state, action, reward, next_state, done)]

        # === 準備 context batch ===
        context = torch.cat([state, action, reward.unsqueeze(1), next_state], dim=-1)  # [B, D]
        context = context.unsqueeze(0).to(self.device)                    # [1, B, D]

        # === Encoder 前向與 Loss ===
        task_probs, logits, z = self.encoder(context)                     # logits: [1, num_tasks], z: [1, latent_dim]
        recon = self.decoder(z)                                           # [1, D]
        recon = recon.expand(self.batch_size, -1)                         # [B, D]
        # 對應的 label 要是長度為 1 的 tensor
        task_label = torch.tensor([task_id], device=self.device)

        loss_cls = F.cross_entropy(logits, task_label)
        loss_recon = F.mse_loss(recon, context.squeeze(0))                # 對應 [1, D] vs [D]
        
        # (選配) embedding L2 regularization
        # loss_reg = (z**2).mean() * lambda_reg

        loss_enc = loss_cls + loss_recon                                  # + loss_reg (若有)

        self.optimizer_encoder.zero_grad()
        loss_enc.backward()
        self.optimizer_encoder.step()

        # === Prepare task_info for actor/critic ===
        task_info = F.one_hot(task_probs.argmax(dim=-1), num_classes=self.num_tasks).float()
        task_info = task_info.expand(self.batch_size, -1).detach()

        # === Critic update ===
        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_state, task_info)
            target_q1, target_q2 = self.critic_target(next_state, next_action, task_info)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            y = reward + self.gamma * (1 - done) * target_q.squeeze(-1)

        q1, q2 = self.critic(state, action, task_info)
        critic_loss = F.mse_loss(q1.squeeze(-1), y) + F.mse_loss(q2.squeeze(-1), y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # === Actor update ===
        new_action, log_prob, _ = self.actor.sample(state, task_info)
        q1_pi, q2_pi = self.critic(state, new_action, task_info)
        actor_loss = (self.alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # === Alpha update ===
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # === Soft update ===
        self.train_step += 1
        if self.train_step % self.update_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    
    def save(self, path_prefix):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        torch.save({'actor': self.actor.state_dict(),}, f"{path_prefix}_model.pth")

    def load(self, path_prefix):
        checkpoint = torch.load(f"{path_prefix}_model.pth", map_location=torch.device('cpu'))
        self.actor.load_state_dict(checkpoint['actor'])

def evaluate_agent(agent, env, episodes=5, context_size=100, random_ctx=True):
    device = agent.device
    # 1) 收集 context transitions
    ctx_states, ctx_actions, ctx_rewards, ctx_nexts, ctx_dones = [], [], [], [], []
    s, _ = env.reset()
    for _ in range(context_size):
        if random_ctx:
            a = env.action_space.sample()
        else:
            # 用当前 policy，也可以选择 deterministic=True
            a = agent.select_action(s, deterministic=True)
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc

        ctx_states.append(torch.tensor(s,   dtype=torch.float32))
        ctx_actions.append(torch.tensor(a,  dtype=torch.float32))
        ctx_rewards.append(torch.tensor(r,  dtype=torch.float32))
        ctx_nexts.append(torch.tensor(s2,   dtype=torch.float32))
        ctx_dones.append(torch.tensor(done, dtype=torch.int32))

        s = s2
        if done:
            s, _ = env.reset()

    # 拼 batch，送 encoder
    ctx_states   = torch.stack(ctx_states).to(device)
    ctx_actions  = torch.stack(ctx_actions).to(device)
    ctx_rewards  = torch.stack(ctx_rewards).unsqueeze(-1).to(device)
    ctx_nexts    = torch.stack(ctx_nexts).to(device)
    # ctx_dones   = torch.stack(ctx_dones).to(device)  # encoder 通常不用 done

    with torch.no_grad():
        ctx_z_all = agent.encoder(ctx_states, ctx_actions, ctx_rewards, ctx_nexts)  # [K, latent_dim]
        avg_z     = ctx_z_all.mean(dim=0, keepdim=True)                            # [1, latent_dim]

    # 2) 用这个 avg_z 去跑完整的 episodes
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            # 直接调用 actor，不再 sample buffer
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
            # 这里我们绕过 select_action 的 buffer sampling，直接用 actor.sample
            with torch.no_grad():
                action, _, _ = agent.actor.sample(state_tensor, avg_z)
            action = action.cpu().numpy()[0]

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
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

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    embedding_dim = 64
    agent = SACAgent(state_dim, action_dim, embedding_dim ,env.action_space)
    # agent.load(load_path, train=True)
    num_episodes = 200
    reward_history = [] 
    warmup_episode = 50
    trained_tasks = []
    task_id = 0
    for name, env in env_list:
        print(f"Training on failure joint = {name}")
        trained_tasks.append((name, env))
        agent.memory.clear()
        task_id  += 1
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
                    agent.train(task_id)
                    
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
