import gym
import pybullet_envs

def create_env(env_name='Walker2DBulletEnv-v0'):
    env = gym.make(env_name)
    return env

if __name__ == "__main__":
    env = create_env()
    state = env.reset()
    print("초기 상태 : ", state)