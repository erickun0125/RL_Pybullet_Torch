# src/test.py
import torch
from environment import create_env
from agent import SACAgent
import time

def test_agent(num_episodes=10):
    env = create_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SACAgent(state_dim, action_dim, device)
    # 학습 후 저장한 체크포인트 파일의 경로를 지정합니다.
    checkpoint_path = "checkpoints/sac_checkpoint_ep450.pth"
    agent.load(checkpoint_path)
    print(f"체크포인트 불러오기: {checkpoint_path}")

    total_reward = 0
    for episode in range(1, num_episodes + 1):
        env.render()
        
        state = env.reset()
        episode_reward = 0
        done = False

        max_steps = env._max_episode_steps if hasattr(env, '_max_episode_steps') else 1000

        for _ in range(max_steps):
            time.sleep(1/120)
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                break

        total_reward += episode_reward
        print(f"[Test] Episode {episode}, Reward: {episode_reward:.2f}")

    avg_reward = total_reward / num_episodes
    print(f"평균 보상: {avg_reward:.2f}")

if __name__ == "__main__":
    test_agent()
