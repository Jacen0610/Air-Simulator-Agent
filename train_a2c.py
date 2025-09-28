import torch
from multi_agent_env import MultiAgentSimEnv
from a2c_agent import A2CAgent # <-- 导入 A2CAgent
import numpy as np

import os
# 为 A2C 创建一个内存类，用于存储每个智能体的轨迹数据
class A2CMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
    
    def to_dict(self):
        # A2C 的 update 函数不需要 logprobs
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones
        }

class A2CConfig:
    """A2C 超参数配置"""
    def __init__(self):
        # 环境与智能体参数
        self.grpc_address = "localhost:50051"
        self.state_dim = 7               # 状态维度
        self.action_dim = 3              # 动作维度

        # 训练过程参数
        self.max_episodes = 1000         # 总共训练的回合数 (增加以保证充分训练)
        self.update_timestep = 2000      # 每隔多少步更新一次网络 (增加以稳定更新)

        # A2C 核心超参数
        self.lr = 0.0003                 # 学习率 (与 PPO 对齐)
        self.gamma = 0.99                # 折扣因子

        # 模型保存与加载的超参数
        self.checkpoint_dir = "checkpoints_a2c"
        self.save_every_episodes = 5    # 每隔多少个 episode 保存一次模型
        self.resume_from_episode = 0     # 设置为 > 0 的数值以从特定 episode 恢复训练

def main():
    # 初始化超参数配置
    config = A2CConfig()

    # 如果模型保存目录不存在，则创建它
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
        print(f"创建模型保存目录: {config.checkpoint_dir}")

    # 初始化环境
    env = MultiAgentSimEnv(config.grpc_address)
    observations = env.reset()
    agent_ids = env.agent_ids

    # 为每个智能体创建一个 A2CAgent 实例
    agents = {agent_id: A2CAgent(config.state_dim, config.action_dim, config.lr, config.gamma) for agent_id in agent_ids}
    
    # 为每个智能体创建内存
    memories = {agent_id: A2CMemory() for agent_id in agent_ids}

    # --- 断点续训逻辑 ---
    start_episode = 1
    if config.resume_from_episode > 0:
        print(f"--- 正在从 Episode {config.resume_from_episode} 恢复训练 ---")
        for agent_id, agent in agents.items():
            checkpoint_path = f"{config.checkpoint_dir}/a2c_agent_{agent_id}_episode_{config.resume_from_episode}.pth"
            if os.path.exists(checkpoint_path):
                agent.load(checkpoint_path)
                print(f"成功加载智能体 {agent_id} 的模型: {checkpoint_path}")
            else:
                print(f"警告: 未找到智能体 {agent_id} 的模型文件，将从头开始训练。路径: {checkpoint_path}")
        start_episode = config.resume_from_episode + 1

    print("开始 A2C 训练...")
    
    timestep_count = 0

    # 训练循环
    for i_episode in range(start_episode, config.max_episodes + 1):
        if i_episode > 1:
            observations = env.reset()
        
        episode_rewards = {agent_id: 0 for agent_id in agent_ids}

        while True:
            timestep_count += 1
            
            # 所有智能体选择动作
            actions_to_send = {}
            for agent_id, obs in observations.items():
                # A2C 的 select_action 返回 action 和 logprob，但 update 函数不需要 logprob
                action, _ = agents[agent_id].select_action(obs)
                
                memories[agent_id].states.append(torch.FloatTensor(obs))
                memories[agent_id].actions.append(torch.tensor(action))

                # [核心修改] 直接使用网络输出的动作 [0, 1, 2]
                actions_to_send[agent_id] = action

            # 在环境中执行动作
            next_observations, rewards, dones, all_done, _ = env.step(actions_to_send)

            # 存储 reward 和 done
            for agent_id in agent_ids:
                memories[agent_id].rewards.append(rewards[agent_id])
                memories[agent_id].dones.append(dones[agent_id])
                episode_rewards[agent_id] += rewards[agent_id]

            observations = next_observations

            # 如果达到更新步数或回合结束，则更新网络
            if timestep_count % config.update_timestep == 0 or all_done:
                for agent_id in agent_ids:
                    if len(memories[agent_id].states) > 0:
                        agents[agent_id].update(memories[agent_id].to_dict())
                        memories[agent_id].clear()

            if all_done:
                break
        
        # --- 定期保存模型逻辑 ---
        if i_episode % config.save_every_episodes == 0:
            print(f"\n--- 正在保存 Episode {i_episode} 的模型 ---")
            for agent_id, agent in agents.items():
                checkpoint_path = f"{config.checkpoint_dir}/a2c_agent_{agent_id}_episode_{i_episode}.pth"
                agent.save(checkpoint_path)
            print(f"--- 模型保存完毕 ---\n")

        avg_reward = np.mean(list(episode_rewards.values()))
        print(f"Episode {i_episode} | 平均奖励: {avg_reward:.2f}")

if __name__ == '__main__':
    main()