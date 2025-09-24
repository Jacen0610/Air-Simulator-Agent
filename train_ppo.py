import torch
from multi_agent_env import MultiAgentSimEnv
from ppo_agent import PPOAgent # <-- 导入 PPOAgent
import numpy as np

# 为 PPO 创建一个内存类，用于存储每个智能体的轨迹数据
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
    
    def to_dict(self):
        return {
            'states': self.states,
            'actions': self.actions,
            'logprobs': self.logprobs,
            'rewards': self.rewards,
            'dones': self.dones
        }

def main():
    ################## 超参数 ##################
    grpc_address = "localhost:50051"
    max_episodes = 24        # 总共训练的回合数
    update_timestep = 12000   # 每隔多少步更新一次网络
    
    # PPO 相关超参数
    state_dim = 6               # 状态维度
    action_dim = 3              # 动作维度
    lr = 0.0003                 # 学习率
    gamma = 0.99                # 折扣因子
    K_epochs = 2                # 更新策略的 epoch 数
    eps_clip = 0.2              # PPO 裁剪范围

    #############################################

    # 初始化环境
    env = MultiAgentSimEnv(grpc_address)
    observations = env.reset()
    agent_ids = env.agent_ids

    # 为每个智能体创建一个 PPOAgent 实例
    agents = {agent_id: PPOAgent(state_dim, action_dim, lr, gamma, K_epochs, eps_clip) for agent_id in agent_ids}
    
    # 为每个智能体创建内存
    memories = {agent_id: PPOMemory() for agent_id in agent_ids}

    print("开始 PPO 训练...")
    
    timestep_count = 0

    # 训练循环
    for i_episode in range(1, max_episodes + 1):
        if i_episode > 1:
            observations = env.reset()
        
        episode_rewards = {agent_id: 0 for agent_id in agent_ids}

        while True:
            timestep_count += 1
            
            # 所有智能体选择动作
            actions_to_send = {}
            for agent_id, obs in observations.items():
                # PPO 的 select_action 返回 action 和 logprob
                action, logprob = agents[agent_id].select_action(obs)
                
                # 存储 state, action, logprob
                memories[agent_id].states.append(torch.FloatTensor(obs))
                memories[agent_id].actions.append(torch.tensor(action))
                memories[agent_id].logprobs.append(logprob)

                # 将网络输出的动作 [0, 1, 2] 映射到环境的动作 [1, 2, 3]
                actions_to_send[agent_id] = action + 1

            # 在环境中执行动作
            next_observations, rewards, dones, all_done, _ = env.step(actions_to_send)

            # 存储 reward 和 done
            for agent_id in agent_ids:
                memories[agent_id].rewards.append(rewards[agent_id])
                memories[agent_id].dones.append(dones[agent_id])
                episode_rewards[agent_id] += rewards[agent_id]

            observations = next_observations

            # 如果达到更新步数或回合结束，则更新网络
            if timestep_count % update_timestep == 0 or all_done:
                for agent_id in agent_ids:
                    if len(memories[agent_id].states) > 0:
                        agents[agent_id].update(memories[agent_id].to_dict())
                        memories[agent_id].clear()

            if all_done:
                break
        
        avg_reward = np.mean(list(episode_rewards.values()))
        print(f"Episode {i_episode} | 平均奖励: {avg_reward:.2f}")

if __name__ == '__main__':
    main()