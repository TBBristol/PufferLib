import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from pufferlib import PufferEnv

class DIAYNVecEnv(PufferEnv):
    """
    A drop-in VecEnv wrapper that replaces the
    rewards produced by the underlying C backend.
    """
    def __init__(self, base_env, skill_sampler=None, num_skills=4, disc_train_interval=50, device="cpu"):
        
        self.env = base_env
        self.discriminator= Discriminator(state_dim = 6 * self.env.num_stacks, num_skills=num_skills).to(device)
        self.disc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-3
        )
        self.device = device
        self.single_obs_space = self.env.single_observation_space
        self.replay_buffer = DIAYNReplayBuffer(200000, self.single_obs_space.shape, num_skills, device)
        self.num_skills = num_skills
        self.step_count = 0
        self.disc_train_interval = disc_train_interval
        self.disc_batch_size = disc_train_interval // 10

        if skill_sampler is None:
            self.skill_sampler = lambda n: np.random.randint(
                0, self.num_skills, size=n, dtype=np.int64
            )
        else:
            self.skill_sampler = skill_sampler

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _diayn_reward(self, obs_batch):
        self.discriminator.eval()
        with torch.no_grad():
            logits = self.discriminator(torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device))  # (n_envs, K)
            log_probs = torch.log_softmax(logits, dim=1)          # log q(z|s)
            idx = torch.arange(len(obs_batch), device=self.device) #This and next line fancy indexing to select log probs of actaul skill used
            log_q = log_probs[idx, torch.as_tensor(self.current_skills, device=self.device)]
            log_p = -math.log(self.num_skills)  #\log p(z) = \log\left(\frac{1}{K}\right) = -\log(K)
            intrinsic = log_q - log_p # logp is -0.6ish for (4)                 

            return intrinsic.cpu().numpy()                        

    def reset(self, *a, **kw):
        self.step_count = 0
        self.current_skills = self.skill_sampler(self.env.num_envs)
        return self.env.reset(*a, **kw)

    def close(self):
        self.env.close()

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions) #obs is batch/ containers/6
        self.replay_buffer.add(obs, self.current_skills.copy())
        diayn_reward = self._diayn_reward(obs)

        if not info:
            info.append({})

        log_dict = info[0]
        if 'diayn_reward' not in log_dict:
            log_dict['diayn_reward'] = 0.0
        log_dict['diayn_reward'] += diayn_reward.mean()


        done_mask = np.logical_or(term, trunc)
        if done_mask.any():
            self.current_skills[done_mask] = self.skill_sampler(done_mask.sum())

      
        if self.step_count % self.disc_train_interval == 0 and self.replay_buffer.full:
           print(f"step {self.step_count}")
           print(f"envtick {self.env.tick}")
           loss = self.train_discriminator()
           if 'discriminator_loss' not in log_dict:
               log_dict['discriminator_loss'] = 0.0
           log_dict['discriminator_loss'] += loss

        self.step_count += 1
        return obs, diayn_reward, term, trunc, info

    def train_discriminator(self):
        obs, skill = self.replay_buffer.sample(self.disc_batch_size, self.device)
        self.discriminator.train()
        logits = self.discriminator(obs)
        loss = torch.nn.functional.cross_entropy(logits, skill)
        self.disc_opt.zero_grad()
        loss.backward()
        self.disc_opt.step()
        return loss.item()



class DIAYNReplayBuffer:
    def __init__(self, buffer_size, obs_shape, num_skills, device):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.device = device
        self.next_idx = 0
        self.obs = torch.zeros(buffer_size, *obs_shape, device=device)
        self.skills = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.full = False
        self.num_skills = num_skills

    def __len__(self):
        return self.buffer_size if self.full else self.next_idx

    def _advance_ptr(self):
        self.next_idx = (self.next_idx + 1) % self.buffer_size
        self.full |= self.next_idx == 0


    def add(self, obs, skill):
        for o, s in zip(obs, skill):
            self.obs[self.next_idx] = torch.as_tensor(o, device=self.device)
            self.skills[self.next_idx] = int(s)        
            self._advance_ptr()

    def sample(self, batch_size, device = 'cpu'):
        assert len(self) >= batch_size
        idxs = np.random.randint(0, len(self), size=batch_size)
        obs   = torch.as_tensor(self.obs[idxs],   dtype=torch.float32, device=device)
        skill = torch.as_tensor(self.skills[idxs], dtype=torch.long,    device=device)
        return obs, skill

class Discriminator(nn.Module):
    """Predicts the skill being used from the state
    state_dim is the dimension of the state space
    num_skills is the number of skills to predict
    Remember to use only state_dim for the discriminator not state with skill
    """
    def __init__(self, state_dim: int=512, num_skills:int=4):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_skills) #output is logits per skill
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = torch.flatten(x, start_dim=1, end_dim=-1) #batch, containers, 6
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits =  self.fc3(x)  #no softmax use with XELoss
        return logits       
