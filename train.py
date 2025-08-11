import torch
from multi_agent_env import MultiAgentSimEnv
from dqn_agent import DQNAgent # <-- 导入 DQNAgent
import numpy as np

def main():
    ################## 超参数 ##################
    grpc_address = "localhost:50051"
    max_episodes = 20        # 总共训练的回合数
    max_timesteps = 10000        # 每个回合的最大步数
    
    # DQN 相关超参数
    state_dim = 6               # 状态维度，根据 AgentObservation 定义
    # 有效动作是 WAIT, SEND_PRIMARY, SEND_BACKUP，共3个
    action_dim = 3              # 动作维度
    lr = 0.0001                  # 学习率
    gamma = 0.99                # 折扣因子
    buffer_size = 10000         # 经验回放缓冲区大小
    batch_size = 64             # 学习时采样的批量大小
    tau = 0.005                 # 目标网络软更新系数

    # Epsilon-Greedy 探索策略的超参数
    epsilon_start = 1.0         # 探索率初始值
    epsilon_end = 0.01          # 探索率最终值
    epsilon_decay = 0.995       # 探索率衰减因子

    #############################################

    # 初始化环境
    env = MultiAgentSimEnv(grpc_address)

    # 第一次调用 reset() 来获取智能体 ID 和第一个 episode 的初始观测值。
    # 这样可以避免在训练循环开始时再次重置。
    observations = env.reset()
    agent_ids = env.agent_ids

    # 为每个智能体创建一个 PPOAgent 实例
    agents = {agent_id: DQNAgent(state_dim, action_dim, lr, gamma, buffer_size, batch_size, tau) for agent_id in agent_ids}

    print("开始训练...")
    epsilon = epsilon_start

    # 训练循环
    for i_episode in range(1, max_episodes + 1):
        # 对于第一个 episode，我们已经有了初始观测值。
        # 对于后续的 episodes，我们才需要重置环境。
        if i_episode > 1:
            observations = env.reset()
        
        episode_rewards = {agent_id: 0 for agent_id in agent_ids}

        for t in range(max_timesteps):
            # 所有智能体选择动作
            actions_to_send = {}
            actions_taken = {}
            for agent_id, obs in observations.items():
                action = agents[agent_id].select_action(obs, epsilon)
                actions_taken[agent_id] = action
                # 将网络输出的动作 [0, 1, 2] 映射到环境的动作 [1, 2, 3]
                actions_to_send[agent_id] = action + 1

            # 在环境中执行动作
            next_observations, rewards, dones, all_done, _ = env.step(actions_to_send)

            # 存储 transition 并更新网络
            for agent_id in agent_ids:
                # 获取该智能体相关的数据
                obs = observations[agent_id]
                action = actions_taken[agent_id]
                reward = rewards[agent_id]
                next_obs = next_observations[agent_id]
                done = dones[agent_id]
                
                # 存入经验回放区并执行一次更新
                agents[agent_id].store_transition(obs, action, next_obs, reward, done)
                agents[agent_id].update()
                episode_rewards[agent_id] += reward

            observations = next_observations

            if all_done:
                break
        
        # 衰减 Epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        avg_reward = np.mean(list(episode_rewards.values()))
        print(f"Episode {i_episode} | 平均奖励: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")

if __name__ == '__main__':
    main()