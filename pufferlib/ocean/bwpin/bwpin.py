'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.bwpin import binding

class Bwpin(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=11, buf=None, seed=0, pegs = 3,max_timesteps = 250, celebrate_ticks=1):
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1,
            shape=((pegs *2)+1,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete((pegs*2 +1) *2) -2)) #each space both directions minus two outsides which jsut go one dir
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.pegs = pegs
        self.max_timesteps = max_timesteps
        if self.pegs < 3:
            raise ValueError("pegs must be >= 3")
        self.celebrate_ticks = celebrate_ticks

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, pegs= self.pegs, max_timesteps=self.max_timesteps, celebrate_ticks = self.celebrate_ticks)
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1

        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)
        
    def set_goal(self, env_id):
        binding.vec_goal_set(self.c_envs, env_id)

    def reverse_trajectory(self, env_id, num_moves):
        """
        Reverses the trajectory of the given environment and retyrbs as tensor[num_moves+1, obs_dim]
        FOR SINGLE ENV
        """
        set_goal(env_id)

        obs_dim - self.single_observation_space.shape[0]
        traj = torch.empty((num_moves+1, obs_dim), dtype=torch.int64)

        traj[0].copy_(torch.from_numpy(self.observations[env_id]).to(torch.int64))
        for step in range(1, num_moves+1):
            binding.vec_shuffle_moves(self.c_envs, env_id, 1)
            traj[step].copy_(torch.from_numpy(self.observations[env_id]).to(torch.int64))

        return traj

    def vec_reverse_trajectory(self, num_moves):
        for i in range(self.num_agents):
            self.set_goal(i)

        obs_dim = self.single_observation_space.shape[0]
        batch_traj = torch.empty((self.num_agents, num_moves+1, obs_dim), dtype=torch.int64)
        batch_traj[:,0,:].copy_(torch.from_numpy(self.observations).to(torch.int64))
        for step in range(1, num_moves+1):
            binding.vec_shuffle_all(self.c_envs, 1)
            batch_traj[:,step,:].copy_(torch.from_numpy(self.observations).to(torch.int64))

        return batch_traj

from pathlib import Path
import torch
import cloudpickle


def collect_reverse_dataset(env, traj_len: int, shard_traj_count: int, num_shards: int, out_dir: str):
      obs_dim = env.single_observation_space.shape[0]
      out_dir = Path(out_dir)
      out_dir.mkdir(parents=True, exist_ok=True)

      shard = torch.empty((shard_traj_count, traj_len + 1, obs_dim), dtype=torch.int64)
      filled = 0
      shard_idx = 0

      while shard_idx < num_shards:
          batch = env.vec_reverse_trajectory(traj_len)  # shape [num_envs, traj_len+1, obs_dim]
          batch = batch.to(dtype=torch.int64)          # already int64, but explicit is fine

          start = 0
          while start < batch.size(0):
              remaining = shard_traj_count - filled
              take = min(remaining, batch.size(0) - start)

              shard[filled:filled + take].copy_(batch[start:start + take])
              filled += take
              start += take

              if filled == shard_traj_count:
                  shard_path = out_dir / f"bwpin_reverse_T{traj_len+1}_obs{obs_dim}_{shard_idx:04d}.pkl"
                  with shard_path.open("wb") as f:
                    cloudpickle.dump(shard.clone(), f)        # torch.save ⇒ pickle file
                  shard_idx += 1
                  filled = 0
                  if shard_idx == num_shards:
                      return  # stop immediately, discard any leftover batch data

      if filled > 0:
          shard_path = out_dir / f"bwpin_reverse_T{traj_len+1}_obs{obs_dim}_{shard_idx:04d}.pkl"
          with shard_path.open("wb") as f:
              cloudpickle.dump(shard[:filled].clone(), f)


if __name__ == '__main__':
    N = 4096

    env = Bwpin(num_envs=N)
    #env.reset()
    # steps = 0

    # CACHE = 1024
    # actions = np.random.randint(0, 6, (CACHE, N))

    # i = 0
    # import time
    # start = time.time()
    # while time.time() - start < 10:
    #     env.step(actions[i % CACHE])
    #     steps += N
    #     i += 1

    # print('Bwpin SPS:', int(steps / (time.time() - start)))
    traj_len = 15
    shard_traj_count = 1500000
    num_shards = 5
    out_dir = "bwpin_reverse"
    collect_reverse_dataset(env, traj_len, shard_traj_count, num_shards, out_dir)
