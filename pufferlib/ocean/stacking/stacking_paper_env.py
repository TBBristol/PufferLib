import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

def generate_container_instance(scale):
    """
    Generate a single container instance of length = scale.
    Returns a list of container 'priorities' (e.g. departure orders).
    You can randomize them or follow your specific distribution.
    """
    # Example: a random permutation of [0..scale-1]
    containers = list(range(scale))
    random.shuffle(containers)
    return containers


class ContainerStackEnv(gym.Env):
    """
    A container stacking environment with an observation of shape (n_stacks, 6).
    Returns 5-tuple from step: (obs, reward, terminated, truncated, info).
    """
    def __init__(self, training_data, max_height, max_steps=None):
        super().__init__()
        self.training_data = training_data
        self.containers = None
        self.num_containers = None
        self.max_height = max_height

        # We'll define placeholders for action/obs space.
        # Because scale can vary if you want, but for simplicity,
        # assume we have one scale across all instances.
        # If scale changes, you need a more dynamic approach.
        
        # We'll figure out how many containers from the first instance
        # (assuming they all have the same scale).
        example_containers = self.training_data[0]
        self.num_containers = len(example_containers)
        
        # Derived number of stacks
        self.n_stacks = math.ceil(self.num_containers / self.max_height)
        
        # Actions: choose which stack to place the next container
        self.action_space = spaces.Discrete(self.n_stacks)
        
        # Observations: shape = (n_stacks, 6)
        self.observation_space = spaces.Box(
            low=0.0,
            high=float('inf'),  # or refine as needed
            shape=(self.n_stacks, 6),
            dtype=np.float32
        )
        
        # Internal state
        self.stacks = None
        self.next_container_index = None
        self.current_unsorted = None
        
        # If you want a time limit, set max_steps (optional)
        # If None, we won't truncate by steps.
        self.max_steps = max_steps
        self.current_step = 0
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(a = seed)
        # Randomly pick one instance from the training set
        self.containers = random.choice(self.training_data)
        self.num_containers = len(self.containers)
        self.n_stacks = math.ceil(self.num_containers / self.max_height)

        # Re-init episode
        self.stacks = [[] for _ in range(self.n_stacks)]
        self.next_container_index = 0
        self.current_unsorted = 0
        self.current_step = 0
        
        obs = self._get_observation()
        info = {}
        return obs, info
    
    def get_action_mask(self):

        possible_actions = np.full(self.n_stacks, False)
        for i in range(self.n_stacks):
            if len(self.stacks[i]) < self.max_height:
                possible_actions[i] = True
        return possible_actions
    
    def action_masks(self):
        """This is named as action_masks in sb3_contrib.common.maskable.utils for compatibility with sb3_contrib.common.maskable.utils.get_action_masks
            Same fn as get_action_mask()
        Returns:
            _type_: _description_
        """
        possible_actions = np.full(self.n_stacks, False)
        for i in range(self.n_stacks):
            if len(self.stacks[i]) < self.max_height:
                possible_actions[i] = True
        return possible_actions

    def step(self, action):
        """
        Return: obs, reward, terminated, truncated, info
        Gymnasium 0.26+ style
        """
        self.current_step += 1
        if type(action) == tuple:
            action = action[0]
        if type(action) == np.ndarray:
            action = int(action)
        # Default
        terminated = False
        truncated = False
        info = {}

        old_unsorted = self.current_unsorted
        
        # check if we have placed all containers, its terminal
        if self.next_container_index >= self.num_containers:
            terminated = True
            return self._get_observation(), 0.0, terminated, truncated, info
        else:
            # Place the next container
            container_priority = self.containers[self.next_container_index]
            self.stacks[action].append(container_priority)
            
            # Recount unsorted
            self.current_unsorted = self._count_unsorted_all_stacks()
            
            # Reward = old_unsorted - new_unsorted
            reward = (old_unsorted - self.current_unsorted) 
          
            
            self.next_container_index += 1
     
        # If there's a max step limit, check if we exceed it
        if self.max_steps is not None and self.current_step >= self.max_steps:
            truncated = True
            # Typically you'd still produce a "final" reward 
            # or partial completion result.
        
        obs = self._get_observation()
        return obs, float(reward), terminated, truncated, info

    def _count_unsorted_all_stacks(self):
        total = 0
        for stack in self.stacks:
            total += self._count_stack_unsorted(stack)
        return total
    
    def _count_stack_unsorted(self, stack):
        count = 0
        for i in range(1, len(stack)):
            if stack[i] < stack[i - 1]:
                count += 1
        return count

    def _get_observation(self):
        """
        Return shape = (n_stacks, 6).
          (1) height%
          (2) next container priority
          (3) top container priority
          (4) lowest remaining priority
          (5) # of remaining containers with priority < top
          (6) # of unsorted in this stack
        """
        obs = np.zeros((self.n_stacks, 6), dtype=np.float32)
        
        # Next container priority
        if self.next_container_index < self.num_containers:
            next_priority = float(self.containers[self.next_container_index])
        else:
            next_priority = 0.0
        
        # Remaining containers
        remaining = self.containers[self.next_container_index:]
        if len(remaining) > 0:
            lowest_priority = float(min(remaining))
        else:
            lowest_priority = 0.0
        
        for i in range(self.n_stacks):
            stack_i = self.stacks[i]
            
            # 1) Height %
            height_percent = len(stack_i) / self.max_height
            
            # 2) Next priority
            next_cont_priority = next_priority
            
            # 3) Top container priority
            if len(stack_i) > 0:
                stack_priority = float(stack_i[-1])
            else:
                stack_priority = 0.0
            
            # 4) Lowest priority of remaining
            low_remaining_prio = lowest_priority
            
            # 5) # of remaining whose priority < top
            count_lower = sum(pr < stack_priority for pr in remaining)
            
            # 6) # of unsorted in this stack
            stack_unsorted = float(self._count_stack_unsorted(stack_i))
            
            obs[i, 0] = height_percent
            obs[i, 1] = next_cont_priority
            obs[i, 2] = stack_priority
            obs[i, 3] = low_remaining_prio
            obs[i, 4] = float(count_lower)
            obs[i, 5] = stack_unsorted
        
        return obs

    def render(self):
        print("=== RENDER ===")
        for i, stack in enumerate(self.stacks):
            print(f"Stack {i}: {stack} (unsorted={self._count_stack_unsorted(stack)})")
        print(f"Next container index = {self.next_container_index}")
        print(f"Total unsorted = {self.current_unsorted}")
        print(f"Step = {self.current_step}")
        print("==============\n")