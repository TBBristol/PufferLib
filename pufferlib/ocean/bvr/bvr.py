import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.bvr import binding

class Bvr(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        framestack=1,
        render_mode=None,
        log_interval=128,
        buf=None,
        seed=0,
    ):
        
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(18 * framestack,),
            dtype=np.float32,
        )

        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )

        self.num_agents = num_envs
        self.framestack = framestack
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.tick = 0

        super().__init__(buf)
        self.actions = self.actions.astype(np.float32)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            num_agents=1,
            framestack=framestack,
        )

    def reset(self, seed=None):
        self.tick = 0
        seed = 0 if seed is None else seed
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.tick += 1

        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

BVR = Bvr

def test_performance(timeout=10, atn_cache=1024):
    env = Bvr(num_envs=1000)
    env.reset()
    tick = 0

    actions = [env.action_space.sample() for _ in range(atn_cache)]

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {env.num_agents * tick / (time.time() - start)}")

if __name__ == "__main__":
    test_performance()
