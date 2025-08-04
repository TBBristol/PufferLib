from pufferlib.ocean import env_creator as base_env_creator
from pufferlib.ocean.stacking.diayn_reward import DIAYNVecEnv
from pufferlib.ocean.environment import MAKE_FUNCTIONS
import functools
import importlib
from pufferlib.ocean import torch

def _build_base(base_make_env, *a, **kw):
    return DIAYNVecEnv(base_make_env(*a, **kw))


def env_creator(name: str, *a, **kw):

    base_make_env = base_env_creator(name, *a, **kw)
    return functools.partial(_build_base, base_make_env)

__all__ = ["env_creator", "torch"]
