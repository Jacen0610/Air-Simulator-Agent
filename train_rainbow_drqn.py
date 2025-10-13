import torch
from single_agent_env import SingleAgentSimEnv
from rainbow_drqn_agent import RainbowDRQNAgent
import numpy as np
import os

def main():
    ################## 超参数 ##################
    grpc_address = "localhost:50051"
    max_episodes = 500
    
    state_dim = 7
    action_dim = 2 # [核心修改] 动作维度现在是 2
    hidden_dim = 64
    lr = 0.0001
    gamma = 0.99
    buffer_size = 100000
    batch_size = 32
    tau = 0.005
    noisy_std = 0.1
    sequence_length = 8
    learning_starts = 10000
    gradient_steps = 1

    checkpoint_dir = "checkpoints"
    save_every_episodes = 10
    resume_from_episode = 0

    #############################################

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"创建模型保存目录: {checkpoint_dir}")

    env = SingleAgentSimEnv(grpc_address)
    agent = RainbowDRQNAgent(state_dim, action_dim, lr, gamma, buffer_size, batch_size, tau, hidden_dim, noisy_std, sequence_length)

    # --- 断点续训逻辑 (单智能体) ---
    start_episode = 1
    if resume_from_episode > 0:
        print(f"--- 正在从 Episode {resume_from_episode} 恢复训练 ---")
        checkpoint_path = f"{checkpoint_dir}/rainbow_drqn_agent_episode_{resume_from_episode}.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            agent.target_net.load_state_dict(checkpoint['policy_net_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"成功加载模型: {checkpoint_path}")
        else:
            print(f"警告: 未找到模型文件，将从头开始训练。路径: {checkpoint_path}")
        start_episode = resume_from_episode + 1

    print("开始 Rainbow DRQN 训练 (单智能体)...")
    
    total_timesteps = 0

    # 训练循环
    for i_episode in range(start_episode, max_episodes + 1):
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
        
        # --- 定期保存模型逻辑 ---
        if i_episode % save_every_episodes == 0:
            print(f"\n--- 正在保存 Episode {i_episode} 的模型 ---")
            checkpoint_path = f"{checkpoint_dir}/rainbow_drqn_agent_episode_{i_episode}.pth"
            torch.save({
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, checkpoint_path)
            print(f"--- 模型保存完毕 ---\n")
        
        if total_timesteps <= learning_starts:
            print(f"Episode {i_episode} | 收集经验中... ({total_timesteps}/{learning_starts})")
        else:
            print(f"Episode {i_episode} | 奖励: {episode_reward:.2f}")

if __name__ == '__main__':
    main()