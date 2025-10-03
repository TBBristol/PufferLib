import brax.envs

BRAX_ENVS = list(brax.envs._envs.keys())
print(BRAX_ENVS)

import functools
import brax.envs
from brax.envs.wrappers.gym import GymWrapper
from brax.envs.wrappers.torch import TorchWrapper
import pufferlib
import pufferlib.emulation
import pufferlib.environments





def single_env_creator(env_name="ant", backend="generalized", **kwargs):
    env = brax.envs.create(env_name=env_name, backend=backend)
    env = GymWrapper(env)
    env = TorchWrapper(env)


    #env = pufferlib.ClipAction(env)  # NOTE: this changed actions space
    #env = pufferlib.EpisodeStats(env)
    env = pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=None)
    return env

def env_creator(env_name="ant", backend="generalized"):
    default_kwargs = {
        "env_name": env_name,
        "backend": backend,
    }
    return functools.partial(single_env_creator, **default_kwargs)


def Brax_Env():
    def __init__(self, name):

        ...

    def reset(self):
        ...
        
    def step(self, action):
        ...
        
    def render(self):
        ...
        
    def close(self):
        ...
        
    def seed(self, seed):
        ...




        
