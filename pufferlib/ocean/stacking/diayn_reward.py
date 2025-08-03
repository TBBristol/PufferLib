import numpy as np
import torch
import math

class DIAYNVecEnv:
    """
    A drop-in VecEnv wrapper that replaces the
    rewards produced by the underlying C backend.
    """
    def __init__(self, vec_env, discriminator, device="cpu"):
        
        self.env = vec_env
        self.discriminator= discriminator.to(device)
        self.device = device

        def __getattr__(self, name):
            return getattr(self.env, name)

        def _diayn_reward(self, obs, raw_reward, skill):
           with torch.no_grad():
            logits = self.net(torch.as_tensor(obs, dtype=torch.float32,
                                              device=self.device)).squeeze(-1)
            logits = logits.cpu().numpy()
            probs = torch.softmax(logits, dim = 1)
            skill_label = skill 
            log_q = torch.log(probs[0, skill_label].clamp(min=1e-8))
            log_p = -math.log(self.num_skills) #\log p(z) = \log\left(\frac{1}{K}\right) = -\log(K)
            reward = log_q - log_p
            return reward

           

        # ---------- thin wrappers ---------- #
        def reset(self, *a, **kw):
            return self.env.reset(*a, **kw)

        def close(self):
            self.env.close()

        def step(self, actions):
            obs, rew, term, trunc, info = self.env.step(actions)
            rew = self._diayn_reward(obs, rew)
            return obs, rew, term, trunc, info
