

import functools

import numpy as np
import gymnasium

import pufferlib
import pufferlib.emulation
import pufferlib.environments
from pufferlib.emulation import GymnasiumPufferEnv

from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
from brax.io import metrics


def single_env_creator(env_name, buf=None, seed=None, **kwargs):
    render_mode = kwargs.pop('render_mode', None) #brax doesnt take this
    seed = kwargs.pop('seed', None)
    buf = kwargs.pop('buf', None)
    env = envs.create(env_name,
                      episode_length=1000,
                      action_repeat=1,
                      auto_reset=True,
                      batch_size=None,
                      backend='spring',
                      **kwargs)
    env = gym_wrapper.GymWrapper(env)
    # automatically convert between jax ndarrays and torch tensors:
    env = GymnasiumCompatWrapper(env)
    env = GymnasiumPufferEnv(env, buf=buf, seed=seed)

    return env


def env_creator(env_name="ant"):
    default_kwargs = {
        "env_name": env_name,
    }
    return functools.partial(single_env_creator, **default_kwargs)


class GymnasiumCompatWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})

    def reset(self, seed=None, options=None):
        obs = self.env.reset(seed)  # Brax returns just obs
        info = {}               # add empty info dict
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Adapt to Gymnasium: return obs, reward, terminated, truncated, info
        terminated = bool(done)
        truncated = bool(info.get("truncation", False))
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

#REFERENCE
 # def create(def create(
 #   env_name: str,
 #   episode_length: int = 1000,
 #   action_repeat: int = 1,
 #   auto_reset: bool = True,
 #   batch_size: Optional[int] = None,
 #   **kwargs,
 # -> Env:
 # """Creates an environment from the registry.

 # Args:
 #   env_name: environment name string
 #   episode_length: length of episode
 #   action_repeat: how many repeated actions to take per environment step
 #   auto_reset: whether to auto reset the environment after an episode is done
 #   batch_size: the number of environments to batch together
 #   **kwargs: keyword argments that get passed to the Env class constructor

 # Returns:
 #   env: an environment
 # """
 # env = _envs[env_name](**kwargs)

 # if episode_length is not None:
 #   env = training.EpisodeWrapper(env, episode_length, action_repeat)
 # if batch_size:
 #   env = training.VmapWrapper(env, batch_size)
 # if auto_reset:
 #   env = training.AutoResetWrapper(env)

 # return env


#def __init__(
#      self,
#      ctrl_cost_weight=0.5,
#      use_contact_forces=False,
#      contact_cost_weight=5e-4,
#      healthy_reward=1.0,
#      terminate_when_unhealthy=True,
#      healthy_z_range=(0.2, 1.0),
#      contact_force_range=(-1.0, 1.0),
#      reset_noise_scale=0.1,
#      exclude_current_positions_from_observation=True,
#      backend='generalized',
#      **kwargs,
#  ):
#
