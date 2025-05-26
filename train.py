import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from utils import Agent
from collections import deque
import wandb


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
    tau = 0.0008
    alpha = 0.2
    lr = 3e-4
    use_wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(state_dim, action_dim, hidden_dim, latent_dim, action_scale, log_std_min, log_std_max, gamma, tau, alpha, device)

    if use_wandb:
        wandb.login()
        wandb.init(project="DRL_Final", name=f"Hyper SAC")
        wandb.watch([agent.encoder, agent.decoder, agent.hypernet])

    ## 我這邊先分開更新，感覺一起
    optimizer_encoder = optim.Adam(list(agent.encoder.parameters()) + list(agent.decoder.parameters()), lr=0.0008)
    optimizer_hyper = optim.Adam(agent.hypernet.parameters(), lr=0.0008)
    num_episodes = 1000
    max_timesteps = 1000
    batch_size = 256
    
    rewards_per_episode = deque([], maxlen=100)
    total_sac_loss = 0.0
    total_enc_dec_loss = 0.0

    for task in range(1):
        env = make_env()
        for episode in tqdm(range(num_episodes)):
            state, _ = env.reset()
            episode_reward = 0
            total_sac_loss = 0.0
            total_enc_dec_loss = 0.0
            num_updates = 0
  
            for t in range(max_timesteps):
                if t < 100:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state, action, reward, next_state)

                next_state, reward, done, truncated, _ = env.step(action)
                agent.replay_buffer.push(state, action, reward, next_state, done or truncated)
                state = next_state
                episode_reward += reward

                if len(agent.replay_buffer) > batch_size:
                    sac_loss, enc_dec_loss = agent.update_actor_critic(agent.replay_buffer, batch_size)
                    num_updates += 1
                    total_sac_loss += sac_loss.item()
                    total_enc_dec_loss += enc_dec_loss.item()

                    optimizer_encoder.zero_grad()
                    enc_dec_loss.backward()
                    optimizer_encoder.step()

                    optimizer_hyper.zero_grad()
                    sac_loss.backward()
                    optimizer_hyper.step()

                    for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
                
                if done or truncated:
                    break
                    
            # episodic update
            if num_updates > 0:
                avg_enc_dec_loss = total_enc_dec_loss / num_updates
                avg_sac_loss     = total_sac_loss     / num_updates
                total_loss         = avg_enc_dec_loss + avg_sac_loss
            else:
                avg_enc_dec_loss = avg_sac_loss = total_loss = 0.0


            rewards_per_episode.append(episode_reward)
            print(f"Episode {episode}: Total_Loss: {total_loss:.2f}, Recon Loss: {total_enc_dec_loss:.2f}, SAC Loss: {total_sac_loss:.2f}, Reward: {np.mean(rewards_per_episode):.2f}")
        # limit the access to the past experiences
            if episode % 500 == 0:
                agent.save("checkpoints/", episode)

        agent.replay_buffer.clear()
if __name__== '__main__':
    main()