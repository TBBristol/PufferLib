import numpy as np
import torch
import math

class DIAYNVecEnv:
    """
    A drop-in VecEnv wrapper that replaces the
    rewards produced by the underlying C backend.
    """
    def __init__(self, vec_env, discriminator, skill_sampler=None, num_skills=4, device="cpu"):
        
        self.env = vec_env
        self.discriminator= discriminator.to(device)
        self.device = device

        if skill_sampler is None:
            self.skill_sampler = lambda n: np.random.randint(
                0, num_skills, size=n, dtype=np.int64
            )
        else:
            self.skill_sampler = skill_sampler

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _diayn_reward(self, obs_batch, skill):
        self.discriminator.eval()
        with torch.no_grad():
            logits = self.discriminator(torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device))  # (n_envs, K)
            log_probs = torch.log_softmax(logits, dim=1)          # log q(z|s)
            idx = torch.arange(len(obs_batch), device=self.device)
            log_q = log_probs[idx, torch.as_tensor(self.current_skills, device=self.device)]
            log_p = -math.log(self.num_skills)  #\log p(z) = \log\left(\frac{1}{K}\right) = -\log(K)
            intrinsic = log_q - log_p                             

            return intrinsic.cpu().numpy()                        

    def reset(self, *a, **kw):
        self.current_skills = self.skill_sampler(self.env.num_envs)
        return self.env.reset(*a, **kw)

    def close(self):
        self.env.close()

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)
        done_mask = np.logical_or(term, trunc)
        if done_mask.any():
            self.current_skills[done_mask] = self.skill_sampler(done_mask.sum())
        rew = self._diayn_reward(obs, rew)
        return obs, rew, term, trunc, info
