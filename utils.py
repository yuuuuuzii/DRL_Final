import torch
import random
import numpy as np
from gym import Wrapper
from collections import deque

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

def compute_cl_metrics(R_mean: np.ndarray,
                       R_std:  np.ndarray,
                       num_eval_episodes: int
                       ) -> dict:
    
    if R_mean.shape != R_std.shape:
        raise ValueError("R_mean, R_std ")
    N = R_mean.shape[0]
    if N < 1:
        raise ValueError("(N >= 1)")

    SEM = R_std / np.sqrt(num_eval_episodes)
    N_float = float(N)

    # ACC
    w_A = 2.0 / (N_float * (N_float + 1.0))
    A_mean = 0.0
    var_A  = 0.0
    for i in range(N):
        for j in range(i + 1):
            A_mean += R_mean[i, j]
            var_A  += SEM[i, j]**2
    A_mean *= w_A
    var_A  *= (w_A**2)
    A_std = float(np.sqrt(var_A))

    # FT
    if N > 1:
        w_FT = 2.0 / (N_float * (N_float - 1.0))
    else:
        w_FT = 0.0
    FT_mean = 0.0
    var_FT  = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            FT_mean += R_mean[i, j]
            var_FT  += SEM[i, j]**2
    FT_mean *= w_FT
    var_FT  *= (w_FT**2)
    FT_std = float(np.sqrt(var_FT)) if N > 1 else 0.0

    # BWT
    if N > 1:
        w_B = 2.0 / (N_float * (N_float - 1.0))
    else:
        w_B = 0.0

    sum_pos = 0.0   
    var_pos = 0.0   
    for i in range(1, N):
        for j in range(i):
            sum_pos += R_mean[i, j]
            var_pos  += SEM[i, j]**2

    sum_neg = 0.0  
    var_neg = 0.0   
    for j in range(N - 1):
        count = (N - 1 - j)
        sum_neg += count * R_mean[j, j]
        var_neg += (count**2) * (SEM[j, j]**2)

    BWT_mean = w_B * (sum_pos - sum_neg)
    var_BWT  = (w_B**2) * (var_pos + var_neg)
    BWT_std  = float(np.sqrt(var_BWT)) if N > 1 else 0.0

    return {
        'A_mean':    float(A_mean),
        'A_std':     float(A_std),
        'BWT_mean':  float(BWT_mean),
        'BWT_std':   float(BWT_std),
    }