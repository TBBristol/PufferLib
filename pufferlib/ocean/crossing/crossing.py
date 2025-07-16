'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np
import pufferlib
from pufferlib.ocean.crossing import binding

class RiverCrossing(pufferlib.PufferEnv):
    def __init__(self, 
                 boats=1,
                  passengers=1, 
                  max_passengers = 1, 
                  max_boats = 1,
                  max_ep_steps = 100,
                  num_envs=1, 
                  render_mode=None, 
                  log_interval=128, 
                  buf=None, 
                  seed=0):
         #OBS space is:
        
        """newobs
        num_entities = max_passengers *2 + max_boats
        num_rows = num_entities
        Each row is:
        [OHE entity type, OHE entity, OHE paired entity, OHE location]
        Size of obs is:
        3 entity types + num_entites + num_entities + 2 loctions (L/R) and + num_boats locations
        """
        num_entities = max_passengers * 2 + max_boats
        obs_shape_length = 3 + num_entities * 2 + 2 + max_boats


        self.single_observation_space = gymnasium.spaces.Box(low = 0, 
                                                             high = 1,
                                                             shape = (num_entities, obs_shape_length),
                                                             dtype=np.int32)
        #Actions space is:
        # [boat_id, passenger/agent_id, action(0=unload, 1=load, 2=move)]
        self.single_action_space = gymnasium.spaces.MultiDiscrete((boats, 2*passengers,3),dtype=np.int32)

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
                                        boats = boats,
                                        passengers= passengers, 
                                        max_passengers= max_passengers,
                                        max_boats = max_boats,
                                        max_ep_steps = max_ep_steps
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
    max_ep_steps = 100
    passengers = 1
    max_passengers = 1
    max_boats = 1
    boats = 1
    env = RiverCrossing(passengers=1, max_boats=max_boats,boats=boats, max_passengers=max_passengers)
    obs, _ = env.reset()
    print(env.action_space)
    print(obs)
    print("STOP")
   

