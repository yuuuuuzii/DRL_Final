import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import gymnasium as gym
from utils import ReplayBuffer, Agent
from torch.nn.utils import vector_to_parameters
from collections import deque
import ipdb
def make_env(task_id=None):
    env = gym.make("HalfCheetah-v4")
    if task_id is not None:
        env.unwrapped._goal_velocity = task_id * 0.3
    return env

def main():
    env = make_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256
    latent_dim = 128
    action_scale = 1.0
    log_std_min = -20
    log_std_max = 2
    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    lr = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(state_dim, action_dim, hidden_dim, latent_dim, action_scale, log_std_min, log_std_max, gamma, tau, alpha, device)

    ## 我這邊先分開更新，感覺一起
    optimizer_encoder = optim.Adam(list(agent.encoder.parameters()) + list(agent.decoder.parameters()), lr=lr)
    optimizer_hyper = optim.Adam(agent.hypernet.parameters(), lr=lr)
    num_episodes = 1000
    max_timesteps = 1000
    batch_size = 256

    rewards_per_episode = deque([], maxlen=100)
    total_sac_loss = 0
    total_enc_dec_loss = 0
    total_timesteps = 0

    for task in range(10):
        env = make_env(task)
        for episode in tqdm(range(num_episodes)):
            state, _ = env.reset()
            episode_reward = 0

            for t in range(max_timesteps):
                if t < 100:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)

                next_state, reward, done, truncated, _ = env.step(action)
                agent.replay_buffer.push(state, action, reward, next_state, done or truncated)
                state = next_state
                episode_reward += reward

                if len(agent.replay_buffer) > batch_size:
                    sac_loss, enc_dec_loss = agent.update_actor_critic(agent.replay_buffer, batch_size)
                
                    total_sac_loss += sac_loss
                    total_enc_dec_loss += enc_dec_loss


                # 還沒用到這東西    
                total_timesteps += 1
            
                if t % 1 == 0:
                    for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
                
                if done or truncated:
                    break
            
            # episodic update
            total_loss = total_enc_dec_loss + total_sac_loss
            optimizer_encoder.zero_grad()
            total_enc_dec_loss.backward()
            optimizer_encoder.step()

            optimizer_hyper.zero_grad()
            total_sac_loss.backward()
            optimizer_hyper.step()
        
        # limit the access to the past experiences
        agent.replay_buffer.clear()

        rewards_per_episode.append(episode_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}: ELoss: {total_loss.item():.2f}, Recon Loss: {enc_dec_loss.item():.2f}, SAC Loss: {sac_loss.item():.2f}, Reward: {np.mean(rewards_per_episode):.2f}")
        
        if episode % 500 == 0:
            agent.save("checkpoints/", episode)

if __name__== '__main__':
    main()
    print("hi")