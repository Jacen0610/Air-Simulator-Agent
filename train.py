import torch
from multi_agent_env import MultiAgentSimEnv
from ppo_agent import PPOAgent
from collections import defaultdict

def main():
    ################## 超参数 ##################
    grpc_address = "localhost:50051"
    max_episodes = 10000        # 总共训练的回合数
    max_timesteps = 300         # 每个回合的最大步数
    update_timestep = 2000      # 每隔多少步更新一次策略
    
    # PPO 相关超参数
    state_dim = 6               # 状态维度，根据 AgentObservation 定义
    action_dim = 4              # 动作维度，根据 Action 枚举 (忽略 UNSPECIFIED)
    lr = 0.002                  # 学习率
    gamma = 0.99                # 折扣因子
    K_epochs = 4                # 更新策略的 epoch 数
    eps_clip = 0.2              # PPO 的裁剪范围
    #############################################

    # 初始化环境
    env = MultiAgentSimEnv(grpc_address)

    # 第一次调用 reset() 来获取智能体 ID 和第一个 episode 的初始观测值。
    # 这样可以避免在训练循环开始时再次重置。
    observations = env.reset()
    agent_ids = env.agent_ids

    # 为每个智能体创建一个 PPOAgent 实例
    agents = {agent_id: PPOAgent(state_dim, action_dim, lr, gamma, K_epochs, eps_clip) for agent_id in agent_ids}
    
    # 用于为每个智能体存储经验的内存
    memory = defaultdict(lambda: {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': []})

    print("开始训练...")
    time_step = 0

    # 训练循环
    for i_episode in range(1, max_episodes + 1):
        # 对于第一个 episode，我们已经有了初始观测值。
        # 对于后续的 episodes，我们才需要重置环境。
        if i_episode > 1:
            observations = env.reset()
        current_ep_reward = {agent_id: 0 for agent_id in agent_ids}

        for t in range(max_timesteps):
            time_step += 1
            
            # 所有智能体选择动作
            actions_to_send = {}
            for agent_id, obs in observations.items():
                # +1 是因为我们的动作从 1 开始 (WAIT, SEND_PRIMARY, etc.)
                action, log_prob = agents[agent_id].select_action(obs)
                actions_to_send[agent_id] = action + 1

                # 存储数据
                mem = memory[agent_id]
                mem['states'].append(torch.FloatTensor(obs))
                mem['logprobs'].append(log_prob)
                mem['actions'].append(torch.tensor(action))

            # 在环境中执行动作
            next_observations, rewards, dones, all_done, _ = env.step(actions_to_send)

            # 保存奖励和 done 标志
            for agent_id in agent_ids:
                mem = memory[agent_id]
                mem['rewards'].append(rewards[agent_id])
                mem['dones'].append(dones[agent_id])
                current_ep_reward[agent_id] += rewards[agent_id]

            observations = next_observations

            # 如果收集到足够的数据，就更新所有智能体
            if time_step % update_timestep == 0:
                print(f"  [Timestep {time_step}] 正在更新所有智能体...")
                for agent_id in agent_ids:
                    agents[agent_id].update(memory[agent_id])
                # 清空内存
                memory.clear()
                time_step = 0

            if all_done:
                break
        
        print(f"Episode {i_episode} | 平均奖励: {sum(current_ep_reward.values())/len(agent_ids):.2f}")

if __name__ == '__main__':
    main()