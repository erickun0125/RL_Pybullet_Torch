# src/utils.py
import matplotlib.pyplot as plt

def plot_rewards(rewards, filename='rewards.png'):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.savefig(filename)
    plt.close()
