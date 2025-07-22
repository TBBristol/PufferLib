'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np
import pufferlib
from pufferlib.ocean.stacking import binding
import math

class ContainerStacking(pufferlib.PufferEnv):
    def __init__(self, 
                 num_containers = 10,
                 max_height = 5,
                 num_stacks = None,
                  max_ep_steps = 100,
                  reward_us = -1,
                  reward_max_breach = -10,
                  reset_max_breach = 1,
                  num_envs=1, 
                  render_mode=None, 
                  log_interval=128, 
                  buf=None, 
                  seed=0):
        """OBS shape = (n_stacks, 6).
          (1) height%
          (2) next container priority
          (3) top container priority
          (4) lowest remaining priority
          (5) # of remaining containers with priority < topqui
          (6) # of unsorted in this stack"""
        
        if not num_stacks:
            num_stacks = math.ceil(num_containers/max_height)
        
        self.single_observation_space = gymnasium.spaces.Box(low=np.tile(np.array([0, 0, 0, 0, 0, 0], dtype=np.float32), (num_stacks, 1)),
                                                             high=np.tile(np.array([
                                                                                    1,                      
                                                                                    num_containers,        
                                                                                    num_containers,        
                                                                                    num_containers,        
                                                                                    num_containers,       
                                                                                    max_height - 1         
                                                                                ], dtype=np.float32), (num_stacks, 1)),
                                                                                dtype=np.float32
)
     
        self.single_action_space = gymnasium.spaces.Discrete(num_stacks)

        self.render_mode = render_mode
        self.log_interval = log_interval
        self.num_agents = num_envs
      
        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, 
                                       self.actions, 
                                       self.rewards,
                                       self.terminals, 
                                       self.truncations,
                                       num_envs,
                                       seed,
                                        num_containers=num_containers,
                                        max_height = max_height,
                                        num_stacks = num_stacks,
                                        max_ep_steps=max_ep_steps,
                                        reward_us = reward_us,
                                        reward_max_breach = reward_max_breach,
                                        reset_max_breach = reset_max_breach,
                                        )
 
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
    env = ContainerStacking(num_envs=1, num_containers=51,max_height = 5, render_mode='human')
    env.reset()
    for _ in range(1000):
        action = env.single_action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            env.reset()