# src/main.py
import torch
from environment import create_env
from agent import SACAgent
from utils import plot_rewards

def main():
    env = create_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = SACAgent(state_dim, action_dim, device)
    num_episodes = 500
    # 환경에 따라 최대 스텝 수 설정 (Walker2D의 경우 env._max_episode_steps 사용)
    max_steps = env._max_episode_steps if hasattr(env, '_max_episode_steps') else 1000
    rewards_history = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward

            agent.update()

            if done:
                break

        rewards_history.append(episode_reward)
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")

    # 학습 보상 시각화 저장
    plot_rewards(rewards_history)

if __name__ == "__main__":
    main()
