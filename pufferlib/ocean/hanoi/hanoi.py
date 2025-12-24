'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.hanoi import binding

class Hanoi(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=11, buf=None, seed=0, pegs = 3,disks=3,
                 max_timesteps = 250, celebrate_ticks=1):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(disks*pegs,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(pegs*(pegs-1))
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.pegs = pegs
        self.disks = disks
        self.max_timesteps = max_timesteps
        if self.pegs < 3:
            raise ValueError("pegs must be >= 3")
        if self.disks < 3:
            raise ValueError("disks must be >= 3")
        self.celebrate_ticks = celebrate_ticks

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, pegs= self.pegs, disks=self.disks, max_timesteps=self.max_timesteps, celebrate_ticks = self.celebrate_ticks)
 
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

if __name__ == '__main__':
    N = 4096

    env = Hanoi(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 6, (CACHE, N))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    print('Hanoi SPS:', int(steps / (time.time() - start)))
