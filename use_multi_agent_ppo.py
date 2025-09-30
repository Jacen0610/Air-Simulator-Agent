import torch
from multi_agent_env import MultiAgentSimEnv
from ppo_agent import PPOAgent
import os
import re

class UsePPOConfig:
    """PPO 使用脚本的超参数配置"""
    def __init__(self):
        # 环境与智能体参数
        self.grpc_address = "localhost:50051"
        self.state_dim = 7               # 状态维度
        self.action_dim = 3              # 动作维度

        # PPO 核心超参数 (推理时部分参数非必需，但为了初始化Agent需要传入)
        self.lr = 0.0003
        self.gamma = 0.99
        self.K_epochs = 4
        self.eps_clip = 0.2

        # 模型加载相关的超参数
        self.checkpoint_dir = "checkpoints_ppo"
        # 设置为 > 0 的数值以从特定 episode 加载模型。
        # 如果设置为 0, 脚本会自动查找最新的模型。
        self.load_from_episode = 0

def find_latest_episode(checkpoint_dir):
    """在检查点目录中查找最新的 episode 编号"""
    if not os.path.isdir(checkpoint_dir):
        return 0
    
    max_episode = 0
    # 正则表达式匹配 "ppo_agent_..._episode_(\d+).pth"
    pattern = re.compile(r'ppo_agent_.*_episode_(\d+)\.pth')
    
    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            episode = int(match.group(1))
            if episode > max_episode:
                max_episode = episode
                
    return max_episode

def main():
    """主函数，用于加载并运行训练好的 PPO 模型"""
    # 初始化配置
    config = UsePPOConfig()

    # --- 确定要加载的 Episode ---
    load_episode = config.load_from_episode
    if load_episode == 0:
        print("正在自动查找最新的模型...")
        latest_episode = find_latest_episode(config.checkpoint_dir)
        if latest_episode == 0:
            print(f"错误: 在 '{config.checkpoint_dir}' 目录中没有找到任何模型文件。请先运行训练脚本。")
            return
        load_episode = latest_episode
    
    print(f"--- 准备加载 Episode {load_episode} 的模型 ---")

    # 初始化环境
    env = MultiAgentSimEnv(config.grpc_address)
    
    # 重置环境以获取初始观测值和真实的智能体ID列表
    print("正在重置环境以获取智能体列表...")
    observations = env.reset()
    agent_ids = list(observations.keys())
    print(f"发现智能体: {agent_ids}")

    # 为每个智能体创建一个 PPOAgent 实例
    agents = {agent_id: PPOAgent(config.state_dim, config.action_dim, config.lr, config.gamma, config.K_epochs, config.eps_clip) for agent_id in agent_ids}

    # --- 加载模型权重 ---
    all_agents_loaded = True
    for agent_id, agent in agents.items():
        checkpoint_path = f"{config.checkpoint_dir}/ppo_agent_{agent_id}_episode_{load_episode}.pth"
        if os.path.exists(checkpoint_path):
            agent.load(checkpoint_path)
            print(f"成功加载智能体 {agent_id} 的模型: {checkpoint_path}")
        else:
            print(f"警告: 未找到智能体 {agent_id} 的模型文件: {checkpoint_path}")
            all_agents_loaded = False

    if not all_agents_loaded:
        print("\n警告: 部分智能体的模型未能加载，运行结果可能不佳。")

    print("\n--- 开始使用加载的模型运行仿真 ---")

    # 运行 5 个回合来展示效果
    for i_episode in range(1, 6):
        if i_episode > 1:
            observations = env.reset()
        
        episode_rewards = {agent_id: 0 for agent_id in agent_ids}
        
        while True:
            actions_to_send = {}
            for agent_id, obs in observations.items():
                action, _ = agents[agent_id].select_action(obs)
                actions_to_send[agent_id] = action

            next_observations, rewards, dones, all_done, _ = env.step(actions_to_send)

            for agent_id in agent_ids:
                if agent_id in rewards:
                    episode_rewards[agent_id] += rewards[agent_id]
            observations = next_observations
            if all_done: break
        
        total_reward = sum(episode_rewards.values())
        print(f"回合 {i_episode} | 总奖励: {total_reward:.2f}")

    print("\n--- 仿真结束 ---")
    env.close()

if __name__ == '__main__':
    main()