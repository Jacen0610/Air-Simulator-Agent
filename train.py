import torch
import numpy as np
import argparse

from single_agent_env import SingleAgentSimEnv

from ppo_rnn_agent import PPORNNAgent, PPOMemory
from rainbow_drqn_agent import RainbowDRQNAgent
from sac_rnn_agent import SACRNNAgent

def train_ppo(agent, env, max_episodes):
    """PPO-RNN 的单智能体训练循环"""
    memory = PPOMemory()
    print("开始 PPO-RNN 训练...")

    for i_episode in range(1, max_episodes + 1):
        observation = env.reset()
        hidden_state = agent.get_initial_hidden_state()
        episode_reward = 0

        while True:
            action, logprob, h_out = agent.select_action(observation, hidden_state)
            
            memory.states.append(torch.FloatTensor(observation))
            memory.actions.append(torch.tensor(action))
            memory.logprobs.append(logprob)

            next_observation, reward, done, _ = env.step(action)

            memory.rewards.append(reward)
            memory.dones.append(done)
            episode_reward += reward

            observation = next_observation
            hidden_state = h_out

            if done:
                if len(memory.states) > 0:
                    agent.update(memory.to_dict())
                    memory.clear()
                break
        
        print(f"Episode {i_episode}/{max_episodes} | 奖励: {episode_reward:.2f}")

def train_off_policy(agent_name, agent, env, max_episodes, learning_starts, gradient_steps):
    """Rainbow-DRQN 和 SAC-RNN 的单智能体训练循环"""
    print(f"开始 {agent_name.upper()}-RNN 训练...")
    total_timesteps = 0

    for i_episode in range(1, max_episodes + 1):
        observation = env.reset()
        hidden_state = agent.get_initial_hidden_state()
        episode_reward = 0
        episode_transitions = []

        while True:
            total_timesteps += 1
            action, h_out = agent.select_action(observation, hidden_state)

            next_observation, reward, done, _ = env.step(action)

            episode_transitions.append((observation, action, reward, done))
            episode_reward += reward

            observation = next_observation
            hidden_state = h_out

            if total_timesteps > learning_starts:
                for _ in range(gradient_steps):
                    agent.update()

            if done:
                agent.store_episode(episode_transitions)
                break
        
        if total_timesteps <= learning_starts:
            print(f"Episode {i_episode}/{max_episodes} | 收集经验中... ({total_timesteps}/{learning_starts})")
        else:
            print(f"Episode {i_episode}/{max_episodes} | 奖励: {episode_reward:.2f}")

def main():
    parser = argparse.ArgumentParser(description="使用指定的强化学习算法训练单个智能体")
    parser.add_argument("--agent", type=str, required=True, choices=["ppo", "rainbow", "sac"], help="要使用的算法 (ppo, rainbow, sac)")
    args = parser.parse_args()

    ################## 通用超参数 ##################
    grpc_address = "localhost:50051"
    max_episodes = 500
    state_dim = 7
    action_dim = 2 # [核心修改] 动作维度现在是 2 (WAIT, SEND)
    hidden_dim = 64
    lr = 0.0003
    gamma = 0.99

    # Off-policy 专用超参数
    buffer_size = 100000
    batch_size = 32
    tau = 0.005
    sequence_length = 8
    learning_starts = 10000
    gradient_steps = 1

    #############################################

    env = SingleAgentSimEnv(grpc_address)
    agent = None

    # --- 根据选择初始化智能体 ---
    if args.agent == "ppo":
        agent = PPORNNAgent(state_dim, action_dim, lr, gamma, K_epochs=4, eps_clip=0.2, hidden_dim=hidden_dim)
        train_ppo(agent, env, max_episodes)

    elif args.agent == "rainbow":
        agent = RainbowDRQNAgent(state_dim, action_dim, lr, gamma, buffer_size, batch_size, tau, hidden_dim, noisy_std=0.1, sequence_length=sequence_length)
        train_off_policy("rainbow", agent, env, max_episodes, learning_starts, gradient_steps)

    elif args.agent == "sac":
        # [核心修改] SAC 的 target_entropy 也需要更新
        agent = SACRNNAgent(state_dim, action_dim, lr, gamma, buffer_size, batch_size, tau, hidden_dim, sequence_length, alpha=0.2)
        train_off_policy("sac", agent, env, max_episodes, learning_starts, gradient_steps)

if __name__ == '__main__':
    main()