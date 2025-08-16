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
        encoded = self.context_encoder(torch.from_numpy(obs))
        breakpoint()
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


class Bandit(nn.Module):
    """
    Bandit takes in a context vector c0 and outputs a code u
    """


    def __init__(self, d_c, d_u=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_c, 128), nn.ReLU(),
            nn.Linear(128, d_u)
        )
    def forward(self, c0):
        u = self.net(c0)              # shape [B, d_u]
        return F.normalize(u, dim=-1) # (optional) stabilize

from torch.distributions import Bernoulli

class BernoulliHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, h):
        logits = self.lin(h).squeeze(-1)      # [B]
        return Bernoulli(logits=logits)       # distribution object

class LoRAHeads(nn.Module):

    """
    choose a rank r,
    d_u is input dim u from Bandit
    m and n out output dims
    forward takes u and returns A and B which area heads if small linear nets 
    shaped to a matrix size [B,m,r] and [B,n,r]
    """
    def __init__(self, d_u, m, n, r=8):
        super().__init__()
        self.A = nn.Linear(d_u, m*r)
        self.B = nn.Linear(d_u, n*r)
        self.r, self.m, self.n = r, m, n
    def forward(self, u):
        A = self.A(u).view(-1, self.m, self.r)
        B = self.B(u).view(-1, self.n, self.r)
        return A, B

def lora_apply(W, x, A, B, alpha=1.0):
    """
    Apply the heads to input to product a low rank weight update
    ΔW(u)= α⋅A(u)B(u)⊤     (rank ≤r)

    """
    # x: [B,n], W:[m,n], A:[B,m,r], B:[B,n,r]
    # compute (W + A B^T) x 
    #which is: (W x) + A (B^T x)
    Wx = x @ W.t()
    Bt_x = torch.einsum('bnr,bn->br', B, x)            # [B,r]
    A_Bt_x = torch.einsum('bmr,br->bm', A, Bt_x)       # [B,m]
    return Wx + alpha * A_Bt_x

class PolicyLoRA(nn.Module):
    """
    Each layer has its own LoRA heads which are applied to update
    the weights of the later by delta W(u) = alpha⋅A(u)B(u)⊤
    Dynamic parameters are updated by gradient descent and total number
    is (m+n)*r
    Basically what we are doing is updating the weights but doing it by adding A B^T to W and updating those matrices which is smaller than the hught W this lets us paramterise W PER SKILL (u output), effectively this means each skill has its own policy params rather than one policy which adjusts weights per skill

    Example of sizing:
    # Base layer: W ∈ ℝ^(128×256), choose rank r = 8
    #
    # LoRA heads output:
    #   A ∈ ℝ^(128×8)
    #   B ∈ ℝ^(256×8)
    #
    # Dynamic per-skill parameters:
    #   128*8 + 256*8 = 3072 values
    #   (vs 128*256 = 32,768 for full W)
    #
    # Forward pass:
    #   t = Bᵀ x   ∈ ℝ^8
    #   y = σ( W x + α · A t + b )
    #
    # This applies a low-rank update (A Bᵀ) x to W x before activation σ.
    """


    def __init__(self, obs_dim, act_dim, d_u=32, r=8):
        super().__init__()
        # base policy layers
        self.l1  = nn.Linear(obs_dim, 128)
        self.l2  = nn.Linear(128, 128)
        self.out = nn.Linear(128, act_dim)
        # LoRA heads for each layer (they take u and output A,B)
        self.h1 = LoRAHeads(d_u=d_u, m=128, n=obs_dim, r=r)
        self.h2 = LoRAHeads(d_u=d_u, m=128, n=128,    r=r)
        
        self.terminal = BernoulliHead(in_dim=128)


    def forward(self, o, u, alpha=1.0):
        # generate low-rank factors from bandit code u
        A1, B1 = self.h1(u)  # for layer 1
        A2, B2 = self.h2(u)  # for layer 2
        # apply LoRA-adapted layers
        h1 = torch.relu(lora_apply(self.l1.weight, o,   A1, B1, alpha, self.l1.bias))
        h2 = torch.relu(lora_apply(self.l2.weight, h1,  A2, B2, alpha, self.l2.bias))
        return self.out(h2), self.terminal(h2)

