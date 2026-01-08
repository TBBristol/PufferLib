import numpy as np


class GoExploreResetWrapper:
    """Wraps a PufferEnv to ensure every episode starts from a deterministic seed."""

    def __init__(self, env, seed_start=0):
        self.env = env
        self.seed_counter = seed_start
        self.current_seed = None

    def _next_seed(self):
        seed = self.seed_counter
        self.seed_counter += 1
        return seed

    def reset(self, seed=None):
        if seed is None:
            seed = self._next_seed()
        self.current_seed = seed
        return self.env.reset(seed=seed)

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)
        done = (np.any(term) if isinstance(term, np.ndarray) else bool(term)) \
            or (np.any(trunc) if isinstance(trunc, np.ndarray) else bool(trunc))
        if done:
            seed = self._next_seed()
            self.env.reset(seed=seed)
            self.current_seed = seed
        return obs, rew, term, trunc, info

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def close(self):
        return self.env.close()
