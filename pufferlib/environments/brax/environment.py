
from pdb import set_trace as T

import functools

import numpy as np
import gymnasium

import pufferlib
import pufferlib.emulation
import pufferlib.environments


from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
from brax.io import metrics


def single_env_creator(env_name):
    env = envs.create(env_name, batch_size=num_envs,
                    episode_length=episode_length,
                    backend='spring')
    env = gym_wrapper.VectorGymWrapper(env)
    # automatically convert between jax ndarrays and torch tensors:
    env = torch_wrapper.TorchWrapper(env, device=device)
    breakpoint()

   return env


def env_creator(env_name="ant"):
    default_kwargs = {
        "env_name": env_name,
    }
    return functools.partial(single_env_creator, **default_kwargs)
