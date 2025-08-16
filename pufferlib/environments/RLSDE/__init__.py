from pufferlib.ocean import env_creator as base_env_creator
from pufferlib.ocean.stacking.RLSDE import RLSDE 
from pufferlib.ocean.environment import MAKE_FUNCTIONS
import functools
import importlib
from pufferlib.ocean import torch
import inspect

def _build_base(base_make_env, *a, **kw):
    
    sig = inspect.signature(RLSDE.__init__)

    rlsde_param_names = {n for n in sig.parameters if n not in ("self", "base_env")}

    rlsde_kwargs = {k: kw.pop(k) for k in list(kw.keys()) if k in rlsde_param_names}

    base_env = base_make_env(*a, **kw)


    return RLSDE(base_env, **rlsde_kwargs)


def env_creator(name: str, *a, **kw):

    base_make_env = base_env_creator(name, *a, **kw)
    return functools.partial(_build_base, base_make_env)

__all__ = ["env_creator", "torch"]

def __getattr__(name):
    if name == "torch":
        from pufferlib.ocean import torch as _torch
        return _torch
    raise AttributeError(name)
