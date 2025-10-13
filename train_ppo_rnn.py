import torch
from single_agent_env import SingleAgentSimEnv
from ppo_rnn_agent import PPORNNAgent, PPOMemory 
import numpy as np

def main():
    ################## 超参数 ##################
    grpc_address = "localhost:50051"
    max_episodes = 500
    
    state_dim = 7
    action_dim = 2 # [核心修改] 动作维度现在是 2
    hidden_dim = 64
    lr = 0.0003
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2

    #############################################

    env = SingleAgentSimEnv(grpc_address)
    agent = PPORNNAgent(state_dim, action_dim, lr, gamma, K_epochs, eps_clip, hidden_dim)
    memory = PPOMemory()

    print("开始 PPO-RNN 训练 (单智能体)...")

    # 训练循环
    for i_episode in range(1, max_episodes + 1):
        observation = env.reset()
        hidden_state = agent.get_initial_hidden_state()
        episode_reward = 0

        while True:
            action, logprob, h_out = agent.select_action(observation, hidden_state)
            
            memory.states.append(torch.FloatTensor(observation))
            memory.actions.append(torch.tensor(action))
            memory.logprobs.append(logprob)

            next_observation, reward, done, _ = env.step(action)

            memory.rewards.append(reward)
            memory.dones.append(done)
            episode_reward += reward

            observation = next_observation
            hidden_state = h_out

            if done:
                if len(memory.states) > 0:
                    agent.update(memory.to_dict())
                    memory.clear()
                break
        
        print(f"Episode {i_episode} | 奖励: {episode_reward:.2f}")

if __name__ == '__main__':
    main()