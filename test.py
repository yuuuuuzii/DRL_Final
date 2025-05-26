import gymnasium as gym

env = gym.make("HalfCheetah-v4")
obs, _ = env.reset()
total_reward = 0
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
        
    total_reward += reward
print(total_reward)