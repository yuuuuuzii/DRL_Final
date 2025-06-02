import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from torch.distributions import Normal
import os
import pickle
import matplotlib.pyplot as plt
from utils import compute_cl_metrics, JointFailureWrapper, ReplayBuffer

# forward
# TASK_ENCODING_TABLE = torch.tensor([
#     [1, 0, 0, 0, 1, 0, 0, 0],  # task 0
#     [0, 1, 0, 0, 0, 1, 0, 0],  # task 1
#     [0, 0, 1, 0, 0, 0, 1, 0],  # task 2
# ], dtype=torch.float32)

# reverse
TASK_ENCODING_TABLE = torch.tensor([
    [0, 0, 1, 0, 0, 0, 1, 0], # task 2
    [0, 1, 0, 0, 0, 1, 0, 0],  # task 1
    [1, 0, 0, 0, 1, 0, 0, 0],   # task 0
], dtype=torch.float32)

# 取得 embedding
def get_task_encoding(task_id: int, batch_size=None, device='cuda'):
    encoding = TASK_ENCODING_TABLE[task_id].to(device)
    if batch_size is not None:
        encoding = encoding.unsqueeze(0).expand(batch_size, -1)  #[B, D]
    return encoding   


class HyperNetwork(nn.Module):
    def __init__(self, latent_dim, trunk_dim=1024):
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, trunk_dim),
            nn.ReLU(),
            nn.Linear(trunk_dim, 2*trunk_dim),
            nn.ReLU(),
        )

        # Actor head outputs
        self.actor_fc1_head     = nn.Linear(2*trunk_dim, 1152)  # 64 * state_dim + 64 (bias)
        self.actor_fc2_head     = nn.Linear(2*trunk_dim, 4160)  # 64 * 64 + 64 (bias)
        self.actor_mean_head    = nn.Linear(2*trunk_dim, 390)   # action_dim * 64 + action_dim
        self.actor_logstd_head  = nn.Linear(2*trunk_dim, 390)

        # Critic Q1 heads
        self.critic_q1_fc1_head = nn.Linear(2*trunk_dim, 1536)  # 64 * (state+action) + 64
        self.critic_q1_fc2_head = nn.Linear(2*trunk_dim, 4160)
        self.critic_q1_fc3_head = nn.Linear(2*trunk_dim, 65)    # 1 * 64 + 1

        # Critic Q2 heads
        self.critic_q2_fc1_head = nn.Linear(2*trunk_dim, 1536)
        self.critic_q2_fc2_head = nn.Linear(2*trunk_dim, 4160)
        self.critic_q2_fc3_head = nn.Linear(2*trunk_dim, 65)

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
        self.hidden_dim = 64
        self.latent_dim = 8
        self.num_tasks = 4

  
        self.hypernet = HyperNetwork(self.latent_dim, self.hidden_dim).to(self.device)
        self.hypernet_target = HyperNetwork(self.latent_dim, self.hidden_dim).to(self.device)
        self.hypernet_target.load_state_dict(self.hypernet.state_dict())
        
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

        # actor_params 跟 critic_params 分開更新
        self.optimizer_hyper_actor  = torch.optim.Adam(actor_params,  lr=8e-4)
        self.optimizer_hyper_critic = torch.optim.Adam(critic_params, lr=8e-4)

    def count_params(self, module):
        return sum(p.numel() for p in module.parameters())
        
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, task_id, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        task_embed = get_task_encoding(task_id, batch_size=1, device=self.device)
        with torch.no_grad():
            # 重建actor係數
            out = self.hypernet(task_embed)
            actor_fc1, actor_fc2, actor_mean, actor_log_std, *_ = [o.squeeze(0) for o in out]

            if deterministic:
                _, _, action = self.actor.sample(state, actor_fc1, actor_fc2, actor_mean, actor_log_std)
            else:
                action, _, _ = self.actor.sample(state, actor_fc1, actor_fc2, actor_mean, actor_log_std)
        return action.detach().cpu().numpy()[0]
    
    def train(self, task_id):
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = [x.to(self.device) for x in (state, action, reward, next_state, done)]

        task_embed = get_task_encoding(task_id, batch_size=1, device=self.device)
        output = self.hypernet(task_embed) # [B, 1]一組填入係數即可
        actor_fc1, actor_fc2, actor_mean, actor_log_std, critic_q11, critic_q12, critic_q13, critic_q21, critic_q22, critic_q23 = [o.squeeze(0) for o in output]

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = [x.to(self.device) for x in (state, action, reward, next_state, done)]
        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_state, actor_fc1, actor_fc2, actor_mean, actor_log_std)
            
            out_target = self.hypernet_target(task_embed)
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

        new_output = self.hypernet(task_embed)
        n_actor_fc1, n_actor_fc2, n_actor_mean, n_actor_log_std, n_critic_q11, n_critic_q12, n_critic_q13, n_critic_q21, n_critic_q22, n_critic_q23 = [o2.squeeze(0) for o2 in new_output]
        new_action, log_prob, _ = self.actor.sample(state, n_actor_fc1, n_actor_fc2, n_actor_mean, n_actor_log_std)
        sna = torch.cat([state, new_action], dim = -1)

        q1_pi, q2_pi = self.critic(sna, n_critic_q11, n_critic_q12, n_critic_q13, n_critic_q21, n_critic_q22, n_critic_q23)
        actor_loss = (self.alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()
        
        # 更新 Actor Head 
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
        torch.save({
            'actor':          self.actor.state_dict(),
            'hypernet':       self.hypernet.state_dict(),
        }, f"{path_prefix}_model.pth")

    def load(self, path_prefix):
        checkpoint = torch.load(f"{path_prefix}_model.pth", map_location=torch.device('cpu'))
        self.actor.load_state_dict(checkpoint['actor'])
        self.hypernet.load_state_dict(checkpoint['hypernet'])


def evaluate_agent(agent, env, task_id, episodes=5):

    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            # deterministic / eval mode
            action = agent.select_action(state, task_id, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)


if __name__ == "__main__":
    
    failed_joints = [(0, 4), (2, 5)]
    env_list = []
    for joint in failed_joints:
        env = gym.make('HalfCheetah-v4')
        env = JointFailureWrapper(env, failed_joint=joint)
        name = f'HalfCheetah_joint{joint}'
        env_list.append((name, env))
    env = gym.make('HalfCheetah-v4')
    env_list.append((f'HalfCheetah_joint_normal', env))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(state_dim, action_dim, env.action_space)

    # for training
    num_episodes = 200
    reward_history = [] 
    warmup_episode = 50
    trained_tasks = []
    task_id = 0

    # for evaluation
    num_eval_episodes = 5 
    N = len(env_list)
    R_mean = np.zeros((N, N), dtype=np.float32)
    R_std  = np.zeros((N, N), dtype=np.float32)

    for i, (name, env) in enumerate(env_list):
        print(f"Training on failure joint = {name}")
        trained_tasks.append((name, env))
        agent.memory.clear()
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                if episode <= warmup_episode:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state, task_id)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.memory.add(state, action, reward, next_state, done)

                if episode > warmup_episode:
                    agent.train(task_id)
                    
                state = next_state
                total_reward += reward

            reward_history.append(total_reward)

            if (episode + 1) % 20 == 0:
                ckpt_dir = f"checkpoints/task_{i}"
                os.makedirs(ckpt_dir, exist_ok=True)
                agent.save(os.path.join(ckpt_dir, f"sac_hypernet_task{i}_ep{episode+1}"))
                avg_reward = np.mean(reward_history[-20:])
                print(f'"Episode {episode + 1}/{num_episodes}, Total reward: {total_reward:.2f}, joint: {name}')

        task_id += 1

        print(f"\n  >> Evaluating on all {N} tasks using the policy AFTER Task {i}")
        for j, (eval_name, eval_env) in enumerate(env_list):
            mean_r, std_r = evaluate_agent(agent, eval_env, j, episodes=num_eval_episodes)
            R_mean[i, j] = mean_r
            R_std[i, j] = std_r
            print(f"     R[{i},{j}] (mean,std) = ({mean_r:.2f}, {std_r:.2f})")
        print("\n")

    # evaluate metrics
    metrics = compute_cl_metrics(R_mean, R_std, num_eval_episodes)
    print(f"A (mean ± std)   = {metrics['A_mean']:.2f} ± {metrics['A_std']:.2f}")
    print(f"BWT (mean ± std) = {metrics['BWT_mean']:.2f} ± {metrics['BWT_std']:.2f}")

    # visualize reward
    with open('reward_history_CL.pkl', 'wb') as f:
        pickle.dump(reward_history, f)
    plt.plot(reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.vlines(x=200, ymin=min(reward_history), ymax=max(reward_history), colors='r', linestyles='dashed', label='Task 1')
    plt.vlines(x=400, ymin=min(reward_history), ymax=max(reward_history), colors='r', linestyles='dashed', label='Task 2')
    plt.vlines(x=600, ymin=min(reward_history), ymax=max(reward_history), colors='r', linestyles='dashed', label='Task 3')
    plt.legend()
    plt.savefig(f'Training Reward_CL.png')