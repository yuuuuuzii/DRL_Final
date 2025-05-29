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
from clfd.imitation_cl.model.hypernetwork import HyperNetwork, TargetNetwork, calc_delta_theta, calc_fix_target_reg, get_current_targets
from torch.distributions import Normal
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

# Define a fixed Normal object
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean



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

class FunctionalDiagGaussian(nn.Module):
    """DiagGaussian implementation using Functional interface so we can update weights via the hnet."""
    def __init__(self, num_inputs, num_outputs):
        super(FunctionalDiagGaussian, self).__init__()
        self.weights = None

    def set_weights(self, weights):
        assert len(weights) == 3
        # dict for clarity
        self.weights = {'fc_weight': weights[0],
                        'fc_bias': weights[1],
                        'logstd_bias': weights[2]
                        }

    def forward(self, x):
        action_mean = F.linear(x, self.weights['fc_weight'], bias=self.weights['fc_bias'])
        action_logstd = self.weights['logstd_bias']
        return FixedNormal(action_mean, action_logstd.exp())


# class HyperNetwork(nn.Module):
#     def __init__(self, latent_dim, hidden_dim):
#         super().__init__()

#         self.actor_fc1 = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2304),
#         )
#         self.actor_fc2 = nn.Sequential(
#             nn.Linear(latent_dim, 2*hidden_dim),
#             nn.ReLU(),
#             nn.Linear(2*hidden_dim, 16512),
#         )
#         self.actor_mean = nn.Sequential(
#             nn.Linear(latent_dim, 2*hidden_dim),
#             nn.ReLU(),
#             nn.Linear(2*hidden_dim, 774),
#         )
#         self.actor_log_std = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 774),
#         )

        # self.critic_q11 = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim), 
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1536), 
        # )
        # self.critic_q12  = nn.Sequential(
        #     nn.Linear(latent_dim, 2*hidden_dim), 
        #     nn.ReLU(),
        #     nn.Linear(2*hidden_dim, 4160), 
        # )
        # self.critic_q13  = nn.Sequential(
        #     nn.Linear(latent_dim, 16), 
        #     nn.ReLU(),
        #     nn.Linear(16, 65), 
        # )

        # self.critic_q21 = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim), 
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1536), 
        # )
        # self.critic_q22  = nn.Sequential(
        #     nn.Linear(latent_dim, 2*hidden_dim), 
        #     nn.ReLU(),
        #     nn.Linear(2*hidden_dim, 4160), 
        # )
        # self.critic_q23  = nn.Sequential(
        #     nn.Linear(latent_dim, 16), 
        #     nn.ReLU(),
        #     nn.Linear(16, 65), 
        # )

    ## 這邊吃encoder task後產生的embedding
    # def forward(self, embedding):
    #     actor_fc1  = self.actor_fc1(embedding)
    #     actor_fc2  = self.actor_fc2(embedding)
    #     actor_mean  = self.actor_mean(embedding)
    #     actor_log_std = self.actor_log_std(embedding)

    #     # critic_q11 = self.critic_q11(embedding)
    #     # critic_q12 = self.critic_q12(embedding)
    #     # critic_q13 = self.critic_q13(embedding)

    #     # critic_q21 = self.critic_q21(embedding)
    #     # critic_q22 = self.critic_q22(embedding)
    #     # critic_q23 = self.critic_q23(embedding)
    #     return actor_fc1, actor_fc2, actor_mean, actor_log_std #, critic_q11, critic_q12, critic_q13, critic_q21, critic_q22, critic_q23
    
class JointFailureWrapper(Wrapper):
    def __init__(self, env, failed_joint):
        if not hasattr(env, 'reward_range'):
            env.reward_range = (-np.inf, np.inf)
        super().__init__(env)
        self.failed_joint = failed_joint

    def step(self, action):
        a = np.array(action, copy=True)
        # a[self.failed_joint[0]] = 0.0
        # a[self.failed_joint[1]] = 0.0
        a[self.failed_joint] = 0.0
        return self.env.step(a)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_space=None, device='cuda'):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

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
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.apply(weights_init_)

    def forward(self, state, action):
        action = torch.tensor(action, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
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
        # self.actor = Actor(state_dim, action_dim, action_space, self.device).to(self.device)
        self.actor = TargetNetwork(n_in=state_dim, n_out=128, hidden_layers=[128, 128], 
                                   no_weights=True, bn_track_stats=False, 
                                   activation_fn=torch.nn.Tanh(), out_fn=torch.nn.Tanh(), device=self.device)
        self.dist = FunctionalDiagGaussian(128, action_dim)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)

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
        self.beta = 0.01  # regularization loss scaling factor

        # hypernet part
        self.hidden_dim = 4096
        self.latent_dim = 1024
        self.output_a_dim = TargetNetwork.weight_shapes(n_in=state_dim, n_out=128, hidden_layers=[128, 128])
        self.output_dims_dist = [[action_dim, 128], [action_dim], [action_dim]]
        self.task_id = 0
        self.tasks_trained = 0
        self.hypernet = HyperNetwork(self.output_a_dim+self.output_dims_dist, 
                                     layers=[self.hidden_dim] * 2, te_dim=8, device=self.device).to(self.device)
        self.hypernet.gen_new_task_emb()
        self.targets = get_current_targets(self.task_id, self.hypernet)

        self.enc_opt = optim.Adam([self.hypernet.get_task_emb(self.task_id)], lr=3e-4)
        self.hyper_opt = optim.Adam(list(self.hypernet.theta), lr=8e-4)

    def count_params(self, module):
        return sum(p.numel() for p in module.parameters())
    
    def reset_critic(self):
        self.critic = Critic(self.actor.state_dim, self.actor.action_dim).to(self.device)
        self.critic_target = Critic(self.actor.state_dim, self.actor.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)

    def add_task(self):
        self.tasks_trained += 1
        self.hypernet.gen_new_task_emb()
        return self.tasks_trained

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        # actor 基本上是個feature extractor，然後 dist 是個分布
        with torch.no_grad():
            generated_weights = self.hypernet(self.task_id)
            self.actor.set_weights(generated_weights[:len(self.output_a_dim)])
            self.dist.set_weights(generated_weights[len(self.output_a_dim):])
            features, _ = self.actor(state)
            dist = self.dist(features)
            if deterministic:
                action = dist.mode
            else:
                action = dist.sample()
        return action.detach().cpu().numpy()[0]

    def train(self):
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = [x.to(self.device) for x in (state, action, reward, next_state, done)]
        generated_weights = self.hypernet(self.task_id)
        self.actor.set_weights(generated_weights[:len(self.output_a_dim)])
        self.dist.set_weights(generated_weights[len(self.output_a_dim):])

        with torch.no_grad():
            features, _ = self.actor(state)
            dist = self.dist(features)
            log_prob = dist.log_prob(action).unsqueeze(-1)
            next_action = self.select_action(next_state, deterministic=False)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob.mean()
            y = reward + self.gamma * (1 - done) * target_q.squeeze(-1).detach()

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1.squeeze(-1), y) + F.mse_loss(q2.squeeze(-1), y)
        
        new_action = self.select_action(state, deterministic=False)
        new_action = torch.tensor(new_action, dtype=torch.float32).to(self.device)
        new_features, _ = self.actor(state)
        log_prob = self.dist(new_features).log_prob(new_action).unsqueeze(-1)
        q1_pi, q2_pi = self.critic(state, new_action)
        actor_loss = (self.alpha * log_prob.mean() - torch.min(q1_pi, q2_pi)).mean()

        loss = critic_loss + actor_loss

        self.enc_opt.zero_grad()
        self.hyper_opt.zero_grad()
        loss.backward(retain_graph=True)
        self.enc_opt.step()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        dTheta = None

        # Find out the candidate change (dTheta) in trainable parameters (theta) of the hnet
        # This function just computes the change (dTheta), but does not apply it
        dTheta = calc_delta_theta(self.hyper_opt, use_sgd_change=False, detach_dt=True)

        # Calculate the regularization loss using dTheta
        # This implements the second part of equation 2
        loss_reg = calc_fix_target_reg(self.hypernet, self.task_id, targets=self.targets, dTheta=dTheta)

        # Multiply the regularization loss with the scaling factor
        loss_reg *= self.beta
        loss_reg.backward()
        self.hyper_opt.step()
        
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
        # torch.save({'actor': self.actor.state_dict(),}, f"{path_prefix}_model.pth")
        torch.save({
            'actor': self.actor.state_dict(),
            'dist': self.dist.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'hypernet': self.hypernet.state_dict(),
        }, f"{path_prefix}_model.pth")

    def load(self, path_prefix):
        # checkpoint = torch.load(f"{path_prefix}_model.pth", map_location=torch.device('cpu'))
        # self.actor.load_state_dict(checkpoint['actor'])
        checkpoint = torch.load(f"{path_prefix}_model.pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.dist.load_state_dict(checkpoint['dist'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.hypernet.load_state_dict(checkpoint['hypernet'])
        

def evaluate_agent(agent, task_id, env, episodes=5):

    rewards = []
    agent.task_id = task_id
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
    # failed_joints     = [(1, 3) , (2, 5), (0, 4)]
    failed_joints = [1, 3, 5]  # HalfCheetah-v4 有 6 個 actuator
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
    warmup_episode = 10
    trained_tasks = []

    for task_id, (name, env) in enumerate(env_list):
        print(f"Training on failure joint = {name}")
        trained_tasks.append((task_id, name, env))
        agent.memory.clear()
        agent.task_id = task_id
        agent.enc_opt = optim.Adam([agent.hypernet.get_task_emb(task_id)], lr=3e-4)
        if agent.tasks_trained < task_id:
            agent.add_task()
            print(f"Adding new task {task_id} to agent")
        
        for episode in tqdm(range(num_episodes)):
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
                    for task_id, test_name, test_env in trained_tasks:
                        mean_r, std_r = evaluate_agent(agent, task_id, test_env, episodes=5)
                        print(f"Velocity: [{test_name}] avg reward: {mean_r:.2f} ± {std_r:.2f}")
