from pufferlib.ocean import env_creator as base_env_creator
from pufferlib.ocean.stacking.diayn_reward import DIAYNVecEnv
from pufferlib.ocean.environment import MAKE_FUNCTIONS
import functools
import importlib
from pufferlib.ocean import torch
import inspect

def _build_base(base_make_env, *a, **kw):
    
    sig = inspect.signature(DIAYNVecEnv.__init__)

    diayn_param_names = {n for n in sig.parameters if n not in ("self", "base_env")}

    diayn_kwargs = {k: kw.pop(k) for k in list(kw.keys()) if k in diayn_param_names}

    base_env = base_make_env(*a, **kw)


    return DIAYNVecEnv(base_env, **diayn_kwargs)


def env_creator(name: str, *a, **kw):

    base_make_env = base_env_creator(name, *a, **kw)
    return functools.partial(_build_base, base_make_env)

__all__ = ["env_creator", "torch"]

def __getattr__(name):
    if name == "torch":
        from pufferlib.ocean import torch as _torch
        return _torch
    raise AttributeError(name)
