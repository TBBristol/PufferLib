import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from pufferlib import PufferEnv
from gymnasium.spaces.utils import flatdim
from gym.spaces import Box


class RLSDE(PufferEnv):
    """
    A drop-in VecEnv wrapper that replaces the
    rewards produced by the underlying C backend.
    """
    def __init__(self, base_env, device="cpu"):
        
        self.env = base_env
        self.num_skills = num_skills
        self.device = device

        self.raw_obs_space = self.env.single_observation_space


    @property #modify this depending on what we are going to need with skills etc
    def single_observation_space(self):
        return self.raw_obs_space 

    def __getattr__(self, name):
        return getattr(self.env, name)
    
    
    def reset(self, *a, **kw):
        self.step_count = 0
        obs, [] = self.env.reset(*a, **kw)
        return obs, []
      

    def close(self):
        self.env.close()

    def step(self, actions):

        obs, rew, term, trunc, info = self.env.step(actions)
        
        if not info:
            info.append({})

        log_dict = info[0]
  
        self.step_count += 1

        return obs, rew, term, trunc, info

