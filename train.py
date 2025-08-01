# C:/.../Air-Simulator-Py-Agent/train.py
import torch
import torch.optim as optim
import time
import os
from torch.utils.tensorboard import SummaryWriter

from environment import GoSimulatorEnv
from model import ActorCritic

# ==============================================================================
#                           超参数配置 (Hyperparameters)
# ==============================================================================

# --- 核心调整：稳定学习的关键 ---
LEARNING_RATE = 1e-4  # 保持较低的学习率
ENTROPY_COEFF = 0.01  # 熵正则化系数，鼓励探索

# --- 基础配置 ---
GAMMA = 0.99
OBSERVATION_DIM = 6
ACTION_DIM = 3

# --- 训练流程控制 ---
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 13000

# --- 模型保存与日志 ---
SAVE_INTERVAL = 10
MODEL_SAVE_DIR = "models"
LOG_DIR = "logs"
CONTINUE_FROM_EPISODE = 0


def train():
    # ==============================================================================
    #                                1. 准备阶段 (Setup Phase)
    # ==============================================================================
    print("🚀 正在初始化环境和智能体...")

    # 1.1. 初始化环境和日志记录器
    env = GoSimulatorEnv()
    writer = SummaryWriter(LOG_DIR)
    print(f"📝 日志将保存在: {LOG_DIR}")

    # 1.2. 重置环境一次，以获取智能体列表
    initial_observations = env.reset()
    if initial_observations is None:
        print("❌ 无法从环境中获取初始状态，训练终止。")
        env.close()
        return

    agent_ids = list(initial_observations.keys())
    print(f"🤖 发现 {len(agent_ids)} 个智能体: {agent_ids}")

    # 1.3. 创建模型
    agents = {
        agent_id: ActorCritic(OBSERVATION_DIM, ACTION_DIM)
        for agent_id in agent_ids
    }

    # 确保模型保存目录存在
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"📂 创建模型保存目录: {MODEL_SAVE_DIR}")

    # 1.4. 明确的逻辑分支：加载模型 或 开始新训练
    if CONTINUE_FROM_EPISODE > 0:
        print(f"\n🔄 模式: 继续训练。尝试从 Episode {CONTINUE_FROM_EPISODE} 加载模型...")
        for agent_id, model in agents.items():
            load_path = os.path.join(MODEL_SAVE_DIR, f"agent_{agent_id}_episode_{CONTINUE_FROM_EPISODE}.pth")
            if os.path.exists(load_path):
                model.load_state_dict(torch.load(load_path))
                model.train()
                print(f"  - ✅ 成功加载智能体 {agent_id} 的模型。")
            else:
                print(f"  - ⚠️ 警告: 找不到智能体 {agent_id} 的模型文件，将使用新初始化的模型。")
    else:
        print("\n✨ 模式: 开始新的训练。")

    # 1.5. 创建优化器
    optimizers = {
        agent_id: optim.Adam(agents[agent_id].parameters(), lr=LEARNING_RATE)
        for agent_id in agent_ids
    }

    print("\n✅ 准备阶段完成，开始训练循环...")
    # ==============================================================================
    #                                2. 训练阶段 (Training Phase)
    # ==============================================================================
    observations = initial_observations
    start_time = time.time()
    for episode in range(CONTINUE_FROM_EPISODE, NUM_EPISODES):
        if episode > CONTINUE_FROM_EPISODE:
            observations = env.reset()
            if observations is None:
                print(f"❌ 在 Episode {episode + 1} 开始时重置失败，训练终止。")
                break

        total_episode_rewards = {agent_id: 0 for agent_id in agents.keys()}
        is_done = False

        # 2.1. 单个 Episode 的 Step 循环
        for step in range(MAX_STEPS_PER_EPISODE):
            actions_to_take = {}
            log_probs = {}
            state_values = {}
            entropies = {}  # <--- 用于存储熵

            # 为每个智能体选择动作
            for agent_id, obs in observations.items():
                action_dist, state_value = agents[agent_id](obs)
                action = action_dist.sample()

                actions_to_take[agent_id] = action.item()
                log_probs[agent_id] = action_dist.log_prob(action)
                state_values[agent_id] = state_value
                entropies[agent_id] = action_dist.entropy()  # <--- 计算熵

            # 与环境交互
            next_observations, rewards, dones, _ = env.step(actions_to_take)

            if next_observations is None:
                is_done = True
                print("❌ Step 失败，提前结束本轮 Episode。")

            if not is_done:
                # 为每个智能体计算损失并更新网络
                for agent_id in agents.keys():
                    _, next_state_value = agents[agent_id](next_observations[agent_id])
                    advantage = rewards[agent_id] + GAMMA * next_state_value - state_values[agent_id]

                    actor_loss = -log_probs[agent_id] * advantage.detach()
                    critic_loss = advantage.pow(2)

                    # =========================================================
                    #               [核心修复] 重新加入熵正则化
                    # =========================================================
                    entropy_loss = -ENTROPY_COEFF * entropies[agent_id]
                    total_loss = actor_loss + critic_loss + entropy_loss
                    # =========================================================

                    optimizers[agent_id].zero_grad()
                    total_loss.backward()
                    optimizers[agent_id].step()

                    total_episode_rewards[agent_id] += rewards[agent_id]

                if any(dones.values()):
                    is_done = True

                observations = next_observations

            if is_done:
                break

            time.sleep(0.5)

        # 2.2. 日志记录和模型保存 (不变)
        avg_reward = sum(total_episode_rewards.values()) / len(agents) if agents else 0
        writer.add_scalar('Reward/Average_Reward', avg_reward, episode)
        for agent_id, reward in total_episode_rewards.items():
            writer.add_scalar(f'Reward/Agent_{agent_id}', reward, episode)

        elapsed_time = time.time() - start_time
        print(f"Episode {episode + 1}/{NUM_EPISODES} | Avg Reward: {avg_reward:.2f} | Time: {elapsed_time:.2f}s")

        if (episode + 1) % SAVE_INTERVAL == 0:
            print(f"\n💾 Episode {episode + 1}: 正在保存模型...")
            for agent_id, model in agents.items():
                save_path = os.path.join(MODEL_SAVE_DIR, f"agent_{agent_id}_episode_{episode + 1}.pth")
                torch.save(model.state_dict(), save_path)
            print(f"✅ 所有模型已保存至 '{MODEL_SAVE_DIR}' 目录。\n")

    env.close()
    writer.close()


if __name__ == '__main__':
    train()