# src/train.py
import torch
from environment import create_env
from agent import SACAgent
from utils import plot_rewards
import os
import time

def train_agent():
    env = create_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = SACAgent(state_dim, action_dim, device)
    num_episodes = 500
    max_steps = env._max_episode_steps if hasattr(env, '_max_episode_steps') else 1000
    rewards_history = []

    # 체크포인트 및 시각화 저장 폴더 생성
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    for episode in range(1, num_episodes + 1):
        if episode % 100 == 0: env.render()
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            if episode % 100 == 0: time.sleep(1/120)
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward

            agent.update()
            if done:
                break

        rewards_history.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")

        # 일정 에피소드마다 체크포인트 저장 및 보상 시각화
        if episode % 50 == 0:
            checkpoint_path = f"checkpoints/sac_checkpoint_ep{episode}.pth"
            agent.save(checkpoint_path)
            plot_filename = f"plots/rewards_ep{episode}.png"
            plot_rewards(rewards_history, filename=plot_filename)
            print(f"체크포인트 저장: {checkpoint_path}, 보상 그래프 저장: {plot_filename}")

    # 최종 모델 저장 및 보상 그래프 저장
    agent.save("checkpoints/sac_final.pth")
    plot_rewards(rewards_history, filename="plots/rewards_final.png")
    print("학습 완료. 최종 모델과 보상 그래프가 저장되었습니다.")

if __name__ == "__main__":
    train_agent()
