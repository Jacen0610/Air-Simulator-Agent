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
NUM_EPISODES = 25
MAX_STEPS_PER_EPISODE = 5000
# =========================================================
#               [核心修改] 新增更新频率控制
# =========================================================
UPDATE_EVERY_STEPS = 500  # 每隔500个step更新一次网络
# =========================================================

# --- 模型保存与日志 ---
SAVE_INTERVAL = 5
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

    # 1.4. 明确的逻辑分支：加载模型 或 开始新训练 (逻辑保持不变)
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

        # =========================================================
        #      [核心修改] 为每个回合初始化经验缓冲区
        # =========================================================
        batch_log_probs = {agent_id: [] for agent_id in agents.keys()}
        batch_state_values = {agent_id: [] for agent_id in agents.keys()}
        batch_rewards = {agent_id: [] for agent_id in agents.keys()}
        batch_entropies = {agent_id: [] for agent_id in agents.keys()}
        # =========================================================

        # 2.1. 单个 Episode 的 Step 循环
        for step in range(MAX_STEPS_PER_EPISODE):
            actions_to_take = {}

            # 为每个智能体选择动作并存储经验
            for agent_id, obs in observations.items():
                action_dist, state_value = agents[agent_id](obs)
                action = action_dist.sample()

                actions_to_take[agent_id] = action.item()

                # 将经验存入缓冲区
                batch_log_probs[agent_id].append(action_dist.log_prob(action))
                batch_state_values[agent_id].append(state_value)
                batch_entropies[agent_id].append(action_dist.entropy())

            # 与环境交互
            next_observations, rewards, dones, _ = env.step(actions_to_take)

            # 累加每一步的奖励到回合总奖励
            for agent_id, r in rewards.items():
                total_episode_rewards[agent_id] += r
                batch_rewards[agent_id].append(r)  # 将奖励也存入缓冲区

            if next_observations is None:
                is_done = True
                print("❌ Step 失败，提前结束本轮 Episode。")
            else:
                observations = next_observations

            if any(dones.values()):
                is_done = True

            # =========================================================
            #      [核心修改] 每 500 步或在回合结束时，执行一次更新
            # =========================================================
            if (step + 1) % UPDATE_EVERY_STEPS == 0 or is_done:
                # 为每个智能体计算损失并更新网络
                for agent_id in agents.keys():
                    # 如果缓冲区为空，则跳过此智能体的更新
                    if not batch_rewards[agent_id]:
                        continue

                    # 计算 N-Step 的回报
                    _, next_state_value = agents[agent_id](observations[agent_id])
                    if is_done:
                        next_state_value = torch.tensor([0.0])  # 如果回合结束，未来价值为0

                    # 从后往前计算折扣回报
                    returns = []
                    discounted_reward = next_state_value
                    for r in reversed(batch_rewards[agent_id]):
                        discounted_reward = r + GAMMA * discounted_reward
                        returns.insert(0, discounted_reward)

                    # 转换成张量
                    returns = torch.stack(returns)
                    log_probs_tensor = torch.stack(batch_log_probs[agent_id])
                    state_values_tensor = torch.cat(batch_state_values[agent_id]).squeeze()
                    entropies_tensor = torch.stack(batch_entropies[agent_id])

                    # 计算优势
                    advantage = returns - state_values_tensor

                    # 计算损失
                    actor_loss = -(log_probs_tensor * advantage.detach()).mean()
                    critic_loss = advantage.pow(2).mean()
                    entropy_loss = -ENTROPY_COEFF * entropies_tensor.mean()
                    total_loss = actor_loss + critic_loss + entropy_loss

                    # 更新网络
                    optimizers[agent_id].zero_grad()
                    total_loss.backward()
                    optimizers[agent_id].step()

                # 更新后清空所有缓冲区，为下一个批次做准备
                batch_log_probs = {agent_id: [] for agent_id in agents.keys()}
                batch_state_values = {agent_id: [] for agent_id in agents.keys()}
                batch_rewards = {agent_id: [] for agent_id in agents.keys()}
                batch_entropies = {agent_id: [] for agent_id in agents.keys()}
            # =========================================================

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