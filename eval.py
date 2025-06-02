import gymnasium as gym
import torch
import numpy as np

from utils import compute_cl_metrics, JointFailureWrapper
from train_continual import SACAgent as SACAgent_Continual, evaluate_agent as eval_continual
from train_hyper import SACAgent as SACAgent_Hyper, evaluate_agent as eval_hyper

if __name__ == "__main__":
    failed_joints = [(0, 4), (2, 5)]
    env_list = []
    for joint in failed_joints:
        base_env = gym.make("HalfCheetah-v4")
        wrapped = JointFailureWrapper(base_env, failed_joint=joint)
        name = f"HalfCheetah_joint{joint}"
        env_list.append((name, wrapped))
    normal_env = gym.make("HalfCheetah-v4")
    env_list.append(("HalfCheetah_joint_normal", normal_env))

    N = len(env_list)
    num_eval_episodes = 5

    R_mean_cont = np.zeros((N, N), dtype=np.float32)
    R_std_cont  = np.zeros((N, N), dtype=np.float32)
    R_mean_hyp  = np.zeros((N, N), dtype=np.float32)
    R_std_hyp   = np.zeros((N, N), dtype=np.float32)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim  = env_list[0][1].observation_space.shape[0]
    action_dim = env_list[0][1].action_space.shape[0]
    action_sp  = env_list[0][1].action_space

    print("============== Evaluate Continual SAC ==============")
    for i in range(N):
        print(f"Load Continual SAC checkpoint after training task_{i}")
        agent = SACAgent_Continual(state_dim, action_dim, action_sp)
        ckpt_path = f"checkpoints/task_{i}/sac_continual2_task{i}_ep200"
        agent.load(ckpt_path)

        for j, (eval_name, eval_env) in enumerate(env_list):
            mean_r, std_r = eval_continual(agent, eval_env, episodes=num_eval_episodes)
            R_mean_cont[i, j] = mean_r
            R_std_cont[i, j]  = std_r
            print(f"  R_cont[{i},{j}] (mean, std) = ({mean_r:.2f}, {std_r:.2f})")
        print()


    print("=========== Evaluate Hyper Network + SAC ===========")
    for i in range(N):
        print(f"Load Hyper SAC checkpoint after training task_{i}")
        agent_h = SACAgent_Hyper(state_dim, action_dim, action_sp)
        ckpt_path = f"checkpoints/task_{i}/sac_hypernet2_task{i}_ep200"
        agent_h.load(ckpt_path)

        for j, (eval_name, eval_env) in enumerate(env_list):
            mean_r, std_r = eval_hyper(agent_h, eval_env, j, episodes=num_eval_episodes)
            R_mean_hyp[i, j] = mean_r
            R_std_hyp[i, j]  = std_r
            print(f"  R_hyp[{i},{j}] (mean, std) = ({mean_r:.2f}, {std_r:.2f})")
        print()
    
    metrics_cont = compute_cl_metrics(R_mean_cont, R_std_cont, num_eval_episodes)
    metrics_hyp = compute_cl_metrics(R_mean_hyp, R_std_hyp, num_eval_episodes)
    print("===============================================")
    print(f"Accuracy (Continual) = {metrics_cont['A_mean']:.2f} ± {metrics_cont['A_std']:.2f}")
    print(f"BWT      (Continual) = {metrics_cont['BWT_mean']:.2f} ± {metrics_cont['BWT_std']:.2f}")
    print(f"\nAccuracy (Hyper) = {metrics_hyp['A_mean']:.2f} ± {metrics_hyp['A_std']:.2f}")
    print(f"BWT      (Hyper) = {metrics_hyp['BWT_mean']:.2f} ± {metrics_hyp['BWT_std']:.2f}")
    print("===============================================")
