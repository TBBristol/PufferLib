import gymnasium as gym
import pufferlib.emulation
import ale_py

if __name__ == '__main__':
    gym.register_envs(ale_py)

    env = gym.make('ALE/Breakout-v5')
    gymnasium_env = pufferlib.GymToGymnasium(env)
    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(gymnasium_env)
    observations, info = puffer_env.reset()
    action = puffer_env.action_space.sample()
    observation, reward, terminal, truncation, info = puffer_env.step(action)
