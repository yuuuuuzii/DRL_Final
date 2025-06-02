import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import os
from utils import compute_cl_metrics, JointFailureWrapper, ReplayBuffer


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
        self.apply(weights_init_)

        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2., dtype=torch.float32).to(device)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2., dtype=torch.float32).to(device)

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
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


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

        self.target_entropy = -action_dim
        self.gamma = 0.99
        self.tau = 0.005
        self.memory = ReplayBuffer(1000000)
        self.batch_size = 256
        self.train_step = 0
        self.update_freq = 1

        # EWC settings
        self.ewc_lambda = 20000
        self.ewc_tasks = []  

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
    # 計算fisher, 最大化保留參數
    def compute_fisher(self, env, sample_size=1000):
        self.actor.eval()
        fishers = {n: torch.zeros_like(p) for n, p in self.actor.named_parameters()}
        for _ in range(sample_size):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            mean, std = self.actor.forward(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            grads = torch.autograd.grad(log_prob, self.actor.parameters())
            for (n, _), g in zip(self.actor.named_parameters(), grads):
                fishers[n] += g.pow(2)
        for n in fishers:
            fishers[n] /= sample_size
        
        params = {n: p.clone().detach() for n, p in self.actor.named_parameters()}
        self.ewc_tasks.append({'params': params, 'fishers': fishers})
        self.actor.train()

    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = [x.to(self.device) for x in (state, action, reward, next_state, done)]

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            y = reward + self.gamma * (1 - done) * target_q.squeeze(-1)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1.squeeze(-1), y) + F.mse_loss(q2.squeeze(-1), y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor update
        new_action, log_prob, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()

        # EWC regularization
        if self.ewc_tasks:
            ewc_reg = torch.tensor(0., device=self.device)
            for task in self.ewc_tasks:
                for name, param in self.actor.named_parameters():
                    fisher = task['fishers'][name]
                    star = task['params'][name]
                    ewc_reg += (fisher * (param - star).pow(2)).sum()
            actor_loss += (self.ewc_lambda / 2) * ewc_reg

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # alpha update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.train_step += 1
        if self.train_step % self.update_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path_prefix):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        torch.save({'actor': self.actor.state_dict()}, f"{path_prefix}_model.pth")

    def load(self, path_prefix):
        ckpt = torch.load(f"{path_prefix}_model.pth", map_location=torch.device('cpu'))
        self.actor.load_state_dict(ckpt['actor'])


def evaluate_agent(agent, env, episodes=5):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total_r = 0
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, r, term, trunc, _ = env.step(action)
            done = term or trunc
            state = next_state
            total_r += r
        rewards.append(total_r)
    return np.mean(rewards), np.std(rewards)



if __name__ == "__main__":
    
    failed_joints = [(0, 4), (2, 5)]
    env_list = []
    for joint in failed_joints:
        env_list.append((f'HalfCheetah_joint{joint}', JointFailureWrapper(gym.make('HalfCheetah-v4'), failed_joint=joint)))
    env_list.append(( 'HalfCheetah_joint_normal', gym.make('HalfCheetah-v4')))

    state_dim = env_list[0][1].observation_space.shape[0]
    action_dim = env_list[0][1].action_space.shape[0]
    agent = SACAgent(state_dim, action_dim, env_list[0][1].action_space)
    
    reward_history = [] 
    num_episodes = 200
    warmup_episode = 50
    task_id = 0

    num_eval_episodes = 5 
    N = len(env_list)
    R_mean = np.zeros((N, N), dtype=np.float32)
    R_std  = np.zeros((N, N), dtype=np.float32)

    for i, (name, env) in enumerate(env_list):
        print(f"Training on task: {name}")
        agent.memory.clear()
        for ep in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                if ep <= warmup_episode:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)
                next_state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                agent.memory.add(state, action, reward, next_state, done)
                if ep > warmup_episode:
                    agent.train()
                state = next_state
                total_reward += reward
            
            reward_history.append(total_reward)
            if (ep + 1) % 20 == 0:
                ckpt_dir = f"checkpoints/task_{i}"
                os.makedirs(ckpt_dir, exist_ok=True)
                agent.save(os.path.join(ckpt_dir, f"sac_ewc_task{i}_ep{ep+1}"))
                avg_reward = np.mean(reward_history[-20:])
                print(f'"Episode {ep + 1}/{num_episodes}, Total reward: {total_reward:.2f}, joint: {name}')
        # After finishing task, compute EWC
        agent.compute_fisher(env)
        print(f"EWC: Consolidated task {name}")

     
        print(f"\n  >> Evaluating on all {N} tasks using the policy AFTER Task {i} \n")
        for j, (eval_name, eval_env) in enumerate(env_list):
            mean_r, std_r = evaluate_agent(agent, eval_env, episodes=num_eval_episodes)
            R_mean[i, j] = mean_r
            R_std[i, j] = std_r
            print(f"     R[{i},{j}] (mean,std) = ({mean_r:.2f}, {std_r:.2f})")
        print("\n")

    # evaluate metrics
    metrics = compute_cl_metrics(R_mean, R_std, num_eval_episodes)
    print(f"A (mean ± std)   = {metrics['A_mean']:.2f} ± {metrics['A_std']:.2f}")
    print(f"BWT (mean ± std) = {metrics['BWT_mean']:.2f} ± {metrics['BWT_std']:.2f}")
