'''A simple sample environment. Use this as a template for your own envs.'''
import sys
sys.path.insert(0, "/Users/ha24583/Documents/GitHub/river_crossing/PufferLib")
import gymnasium
import numpy as np
import pufferlib
from pufferlib.ocean.crossing import binding

class RiverCrossing(pufferlib.PufferEnv):
    def __init__(self, 
                 boats=1,
                  passengers=3, 
                  max_passengers = 3, 
                  max_boats = 2,
                  max_ep_steps = 100,
                  num_envs=1, 
                  render_mode=None, 
                  log_interval=128, 
                  buf=None, 
                  seed=0):
         #OBS space is:
        """
        [
        [num_passengers, num_agents,num_boats],
        [leftbank[0-1], boat_id, rightbank [0-1]], #for each passenger and agent
        .
        .  2* passengers (equal to agents)   
        . 
        [leftbank[0-1], 0 , rightbank [0-1]].    #for each boat
        .
        .
        [padding, padding, padding] to max_passengers *2 + max_boats
        ]
        """

        num_rows = 1 + max_passengers * 2 + max_boats

        #default low to zero high to zero
        low = np.zeros((num_rows,3), dtype = np.int32)
        high = np.zeros((num_rows,3), dtype = np.int32)

        #These can't take any other values
        #first row first two columns are same value and == number of passengers (equal to num agents)
        high[0,0:2] = max_passengers
        low[0,0:2] = max_passengers
        #first row last column max is num boats
        high[0,2] = max_boats
        low[0,2] = max_boats
        
        #left and right columns are 1 or zero so change the high this is the same for boats
        high[1:max_passengers*2 + 1 + max_boats,0] = 1
        high[1:max_passengers*2 + 1 +max_boats, 2] = 1

        #middle column is boat id for the passenger and agent section so max is boats will start boat ids at 1 so differentiate from not in a boat
        #boats middle column is un-unsed
        high[1:max_passengers*2 + 1, 1] = max_boats


        """
         self.single_observation_space for 3 passengers 1 boat
        Box([[3 3 1]
        [0 0 0]
        [0 0 0]
        [0 0 0]
        [0 0 0]
        [0 0 0]
        [0 0 0]
        [0 0 0]], 

        [[3 3 1]
        [1 1 1]
        [1 1 1]
        [1 1 1]
        [1 1 1]
        [1 1 1]
        [1 1 1]
        [1 0 1]], (8, 3), uint8)
        """
        





        self.single_observation_space = gymnasium.spaces.Box(low =low, 
                                                             high = high, 
                                                             dtype=np.int32)
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
    env = RiverCrossing(passengers=passengers, max_boats=max_boats,boats=boats, max_passengers=max_passengers)
    obs, _ = env.reset()
    print(env.action_space)
    print(obs)
   

