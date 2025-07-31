# C:/.../Air-Simulator-Py-Agent/train.py
import torch
import torch.optim as optim
import time
import os

from environment import GoSimulatorEnv
from model import ActorCritic

# --- 超参数 ---
OBSERVATION_DIM = 6
ACTION_DIM = 3
LEARNING_RATE = 0.001
GAMMA = 0.99

NUM_EPISODES = 50
MAX_STEPS_PER_EPISODE = 13000

# --- 模型保存与加载相关的超参数 ---
SAVE_INTERVAL = 10
MODEL_SAVE_DIR = "models"
CONTINUE_FROM_EPISODE = 0


def train():
    # 1. 初始化环境 (现在这步非常快，只建立连接)
    env = GoSimulatorEnv()

    # 确保模型保存目录存在
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"📂 创建模型保存目录: {MODEL_SAVE_DIR}")

    # 2. **[核心修改]** 在训练开始前，我们还不知道有哪些智能体
    agents = {}
    optimizers = {}

    # 3. 主训练循环
    for episode in range(CONTINUE_FROM_EPISODE, NUM_EPISODES):
        # 3.1. 在每个回合开始时重置环境
        # 这是获取智能体列表和初始观测的唯一地方
        observations = env.reset()
        if observations is None:
            print("❌ 无法从环境中重置，训练终止。")
            break

        # 3.2. **[核心修改]** 如果是第一个回合，现在我们知道了智能体列表，可以创建模型了
        if episode == 0 and not agents:
            agent_ids = list(observations.keys())
            print(f"🤖 发现 {len(agent_ids)} 个智能体: {agent_ids}")

            agents = {
                agent_id: ActorCritic(OBSERVATION_DIM, ACTION_DIM)
                for agent_id in agent_ids
            }

            # 如果是继续训练，则加载模型
            if CONTINUE_FROM_EPISODE > 0:
                print(f"\n🔄 尝试从 Episode {CONTINUE_FROM_EPISODE} 继续训练...")
                # ... (加载模型的逻辑不变) ...
                for agent_id, model in agents.items():
                    load_path = os.path.join(MODEL_SAVE_DIR, f"agent_{agent_id}_episode_{CONTINUE_FROM_EPISODE}.pth")
                    if os.path.exists(load_path):
                        model.load_state_dict(torch.load(load_path))
                        model.train()
                        print(f"  - ✅ 成功加载智能体 {agent_id} 的模型。")
                    else:
                        print(f"  - ⚠️ 警告: 找不到智能体 {agent_id} 的模型文件。")

            optimizers = {
                agent_id: optim.Adam(agents[agent_id].parameters(), lr=LEARNING_RATE)
                for agent_id in agent_ids
            }

        total_episode_rewards = {agent_id: 0 for agent_id in agents.keys()}
        is_done = False

        # 3.3. 单个 Episode 的 Step 循环
        for step in range(MAX_STEPS_PER_EPISODE):
            # ... (内部的 step 逻辑完全不变) ...
            actions_to_take = {}
            log_probs = {}
            state_values = {}

            for agent_id, obs in observations.items():
                action_dist, state_value = agents[agent_id](obs)
                action = action_dist.sample()
                actions_to_take[agent_id] = action.item()
                log_probs[agent_id] = action_dist.log_prob(action)
                state_values[agent_id] = state_value

            next_observations, rewards, dones, _ = env.step(actions_to_take)

            if next_observations is None:  # 检查 gRPC 错误
                is_done = True
                print("❌ Step 失败，提前结束本轮 Episode。")

            if not is_done:
                for agent_id in agents.keys():
                    _, next_state_value = agents[agent_id](next_observations[agent_id])
                    advantage = rewards[agent_id] + GAMMA * next_state_value - state_values[agent_id]
                    actor_loss = -log_probs[agent_id] * advantage.detach()
                    critic_loss = advantage.pow(2)
                    total_loss = actor_loss + critic_loss
                    optimizers[agent_id].zero_grad()
                    total_loss.backward()
                    optimizers[agent_id].step()
                    total_episode_rewards[agent_id] += rewards[agent_id]

                if any(dones.values()):
                    print(f"🏁 Episode {episode + 1} 在第 {step + 1} 步由环境报告完成。")
                    is_done = True

                observations = next_observations

            if is_done:
                break

            time.sleep(0.5)

        # ... (保存模型和打印奖励的逻辑不变) ...
        if (episode + 1) % SAVE_INTERVAL == 0:
            print(f"\n💾 Episode {episode + 1}: 正在保存模型...")
            for agent_id, model in agents.items():
                save_path = os.path.join(MODEL_SAVE_DIR, f"agent_{agent_id}_episode_{episode + 1}.pth")
                torch.save(model.state_dict(), save_path)
            print(f"✅ 所有模型已保存至 '{MODEL_SAVE_DIR}' 目录。\n")

        avg_reward = sum(total_episode_rewards.values()) / len(agents) if agents else 0
        print(f"Episode {episode + 1}/{NUM_EPISODES}, Average Reward: {avg_reward:.2f}")

    env.close()


if __name__ == '__main__':
    train()
