import torch
from multi_agent_env import MultiAgentSimEnv
from rainbow_dqn_agent import RainbowDQNAgent # <-- 导入 RainbowDQNAgent
import numpy as np
import os

def main():
    ################## 超参数 ##################
    grpc_address = "localhost:50051"
    max_episodes = 500       # 总共训练的回合数 (必须增加以进行公平比较)
    max_timesteps = 10000      # 每个回合的最大步数
    
    # Rainbow DQN 相关超参数
    state_dim = 6               # 状态维度
    action_dim = 3              # 动作维度
    lr = 0.0001                 # 学习率
    gamma = 0.99                # 折扣因子
    buffer_size = 10000         # 经验回放缓冲区大小
    batch_size = 64             # 学习时采样的批量大小
    tau = 0.005                 # 目标网络软更新系数
    noisy_std = 0.1             # Noisy Nets 的初始噪声标准差 (关键参数)
    # 在开始学习前，先收集1000步的经验。原8000步的设置导致智能体在整个第一回合都不学习。
    learning_starts = 1000
    gradient_steps = 1          # 每次与环境交互后，执行多少次梯度更新。1是标准做法。

    # 模型保存与加载的超参数
    checkpoint_dir = "checkpoints"    # 模型保存的文件夹
    save_every_episodes = 10          # 每隔多少个 episode 保存一次模型
    resume_from_episode = 0           # 设置为 > 0 的数值以从特定 episode 恢复训练, e.g., 20

    # 注意：Rainbow DQN 使用 Noisy Nets 进行探索，不再需要 Epsilon-Greedy 参数

    #############################################

    # 如果模型保存目录不存在，则创建它
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"创建模型保存目录: {checkpoint_dir}")

    # 初始化环境
    env = MultiAgentSimEnv(grpc_address)

    observations = env.reset()
    agent_ids = env.agent_ids

    # 为每个智能体创建一个 RainbowDQNAgent 实例
    agents = {agent_id: RainbowDQNAgent(state_dim, action_dim, lr, gamma, buffer_size, batch_size, tau, noisy_std) for
              agent_id in agent_ids}

    # --- 断点续训逻辑 ---
    start_episode = 1
    if resume_from_episode > 0:
        print(f"--- 正在从 Episode {resume_from_episode} 恢复训练 ---")
        for agent_id, agent in agents.items():
            checkpoint_path = f"{checkpoint_dir}/rainbow_agent_{agent_id}_episode_{resume_from_episode}.pth"
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                agent.target_net.load_state_dict(checkpoint['policy_net_state_dict']) # 确保 target_net 也同步
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"成功加载智能体 {agent_id} 的模型: {checkpoint_path}")
            else:
                print(f"警告: 未找到智能体 {agent_id} 的模型文件，将从头开始训练。路径: {checkpoint_path}")
        start_episode = resume_from_episode + 1
        # 注意: total_timesteps 不在这里恢复，它会从 0 开始重新计数，但这不影响模型学习。

    print("开始 Rainbow DQN 训练...")
    
    total_timesteps = 0 # 新增：全局步数计数器

    # 训练循环
    for i_episode in range(start_episode, max_episodes + 1):
        if i_episode > 1:
            observations = env.reset()
        
        episode_rewards = {agent_id: 0 for agent_id in agent_ids}

        for t in range(max_timesteps):
            total_timesteps += 1

            # 所有智能体选择动作
            actions_to_send = {}
            actions_taken = {}
            for agent_id, obs in observations.items():
                # select_action 不再需要 epsilon
                action = agents[agent_id].select_action(obs)
                actions_taken[agent_id] = action
                # 将网络输出的动作 [0, 1, 2] 映射到环境的动作 [1, 2, 3]
                actions_to_send[agent_id] = action + 1

            # 在环境中执行动作
            next_observations, rewards, dones, all_done, _ = env.step(actions_to_send)

            # 存储 transition 并更新网络
            for agent_id in agent_ids:
                obs = observations[agent_id]
                action = actions_taken[agent_id]
                reward = rewards[agent_id]
                next_obs = next_observations[agent_id]
                done = dones[agent_id]
                
                agents[agent_id].store_transition(obs, action, next_obs, reward, done)
                
                # 只有在收集到足够多的经验后才开始学习，并且达到更新频率
                if total_timesteps > learning_starts:
                    # 执行 N 次梯度更新，以提高样本利用率
                    for _ in range(gradient_steps):
                        agents[agent_id].update()

                episode_rewards[agent_id] += reward

            observations = next_observations

            if all_done:
                break
        
        # --- 定期保存模型逻辑 ---
        if i_episode % save_every_episodes == 0:
            print(f"\n--- 正在保存 Episode {i_episode} 的模型 ---")
            for agent_id, agent in agents.items():
                checkpoint_path = f"{checkpoint_dir}/rainbow_agent_{agent_id}_episode_{i_episode}.pth"
                torch.save({
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                }, checkpoint_path)
            print(f"--- 模型保存完毕 ---\n")
        
        avg_reward = np.mean(list(episode_rewards.values()))
        
        if total_timesteps <= learning_starts:
            print(f"Episode {i_episode} | 收集经验中... ({total_timesteps}/{learning_starts})")
        else:
            print(f"Episode {i_episode} | 平均奖励: {avg_reward:.2f}")

if __name__ == '__main__':
    main()