import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from pufferlib import PufferEnv
from gymnasium.spaces.utils import flatdim, flatten_space
from gym.spaces import Box
import pdb

class RLSDE(PufferEnv):
    """
    A drop-in VecEnv wrapper that replaces the
    rewards produced by the underlying C backend.
    """
    def __init__(self, base_env, device="cpu"):
        
        self.env = base_env
        
        self.device = device

        self.raw_obs_space = self.env.single_observation_space
        breakpoint()
        
        self.context_encoder = ContextEncoder(self.single_observation_space.shape[0])

    @property #modify this depending on what we are going to need with skills etc
    def single_observation_space(self):
        return flatten_space(self.raw_obs_space)

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


class ContextEncoder(nn.Module):

    """
    A simple MLP to encode the context
    """
    def __init__(self, state_dim, d=64):
        super().__init__()
        self.state_dim = state_dim
        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, d)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.normalize(x, dim=-1)

