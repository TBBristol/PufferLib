import hashlib
import time

import numpy as np
import multiprocessing
import torch
from torch import nn

import pufferlib
import pufferlib.spaces as pspaces


_STATE_ENCODER = None
_KEY_DECIMALS = 4


def _add_cell_unlocked(cells, cell):
    key = cell["key"]
    stored = cells.get(key)

    if stored is None:
        cells[key] = cell
        return True, cell

    if prefer_highest_reward(stored, cell):
        stored["visits"] += 1
        stored["updated_at"] = time.time()
        cell["visits"] = stored["visits"]
        cell["selected"] = 0
        cell["discoveries"] = 0
        cell["created_at"] = stored["created_at"]
        cell["updated_at"] = time.time()
        cells[key] = cell
        return True, cell

    stored["visits"] += 1
    stored["updated_at"] = time.time()
    return False, stored


def _best_cell_unlocked(cells, score_fn=None):
    if not cells:
        return None

    if score_fn is None:
        score_fn = lambda cell: cell["cumulative_reward"]

    return max(cells.values(), key=score_fn)


class CellStore:
    """Container for Go-Explore cells with optional locking."""

    def __init__(self, cells=None, lock=None):
        self.cells = cells or {}
        self.lock = lock

    def __len__(self):
        return len(self.cells)

    def __contains__(self, key):
        return key in self.cells

    def get(self, key, default=None):
        return self.cells.get(key, default)

    def values(self):
        return self.cells.values()

    def items(self):
        return self.cells.items()

    def keys(self):
        return self.cells.keys()

    def copy_cells(self):
        return dict(self.cells)

    def add_cell(self, cell):
        if self.lock:
            with self.lock:
                return _add_cell_unlocked(self.cells, cell)
        return _add_cell_unlocked(self.cells, cell)

    def best(self, score_fn=None):
        if self.lock:
            with self.lock:
                return _best_cell_unlocked(self.cells, score_fn)
        return _best_cell_unlocked(self.cells, score_fn)

    def sample(self, sampler):
        if self.lock:
            with self.lock:
                return sampler(self.cells)
        return sampler(self.cells)


class SharedCellStore(CellStore):
    """Process-safe cell store using multiprocessing.Manager."""

    def __init__(self):
        manager = multiprocessing.Manager()
        cells = manager.dict()
        lock = manager.RLock()
        super().__init__(cells=cells, lock=lock)


def make_env_state(seed=None, extras=None):
    if extras is None:
        extras = {}
    return {"seed": seed, "extras": extras}


def make_cell(key, observation, env_state=None, reward=0.0, trajectory=(), metadata=None):
    if metadata is None:
        metadata = {}

    timestamp = time.time()
    return {
        "key": key,
        "cell_observation": observation,
        "env_state": env_state,
        "cumulative_reward": reward,
        "trajectory_length": len(trajectory),
        "trajectory": tuple(trajectory),
        "metadata": metadata,
        "created_at": timestamp,
        "updated_at": timestamp,
        "visits": 1,
        "selected": 0,
        "discoveries": 0,
    }


def prefer_highest_reward(existing, candidate, reward_eps=1e-6):
    if candidate["cumulative_reward"] > existing["cumulative_reward"] + reward_eps:
        return True

    if abs(candidate["cumulative_reward"] - existing["cumulative_reward"]) <= reward_eps:
        return candidate["trajectory_length"] < existing["trajectory_length"]

    return False


def make_cell_store():
    return CellStore()


def make_shared_cell_store():
    return SharedCellStore()


def add_cell(store, cell):
    if isinstance(store, CellStore):
        return store.add_cell(cell)
    return _add_cell_unlocked(store["cells"], cell)


def get_cell(store, key):
    if isinstance(store, CellStore):
        return store.get(key)
    return store["cells"][key]


def best_cell(store, score_fn=None):
    if isinstance(store, CellStore):
        return store.best(score_fn)
    return _best_cell_unlocked(store["cells"], score_fn)


def reshape_per_env(array, num_envs, agents_per_env):
    arr = np.asarray(array)
    if arr.size == 0:
        return arr

    new_shape = (num_envs, agents_per_env) + arr.shape[1:]
    return arr.reshape(new_shape)


def _resolve_seeds(num_envs, base_seed):
    if base_seed is None:
        return [None for _ in range(num_envs)]
    return [base_seed + idx for idx in range(num_envs)]


def _get_env_seed(tracker, env_idx):
    vecenv = tracker["vecenv"]
    env_obj = None
    if hasattr(vecenv, "envs"):
        envs = vecenv.envs
        if env_idx < len(envs):
            env_obj = envs[env_idx]
    if env_obj is None:
        env_obj = vecenv
    return getattr(env_obj, "current_seed", None)


def _reset_episode(tracker, env_idx, seed):
    tracker["episode_returns"][env_idx] = 0.0
    tracker["step_counts"][env_idx] = 0
    tracker["trajectories"][env_idx] = []
    tracker["episode_seeds"][env_idx] = seed
    tracker["last_cell_key"][env_idx] = None


class InverseDynamicsEncoder:
    """Latent encoder trained via an inverse-dynamics objective (Pathak et al., 2017)."""

    def __init__(
        self,
        observation_shape,
        action_space,
        latent_dim=128,
        hidden_dim=256,
        lr=1e-3,
        device=None,
    ):
        obs_dim = int(np.prod(observation_shape))
        if obs_dim <= 0:
            raise pufferlib.APIUsageError("Observation shape must contain at least one element")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.device = torch.device(device)

        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        ).to(self.device)

        self.action_type, self.action_dim, self.loss_fn = self._configure_action_head(action_space)
        self.inverse_model = nn.Sequential(
            nn.Linear(self.latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        ).to(self.device)

        params = list(self.encoder.parameters()) + list(self.inverse_model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _configure_action_head(self, action_space):
        if isinstance(action_space, pspaces.Discrete):
            action_dim = int(action_space.n)
            loss_fn = nn.CrossEntropyLoss()
            return "discrete", action_dim, loss_fn

        if isinstance(action_space, pspaces.MultiDiscrete):
            action_dim = len(action_space.nvec)
            loss_fn = nn.MSELoss()
            return "multidiscrete", action_dim, loss_fn

        if isinstance(action_space, pspaces.Box):
            action_dim = int(np.prod(action_space.shape))
            loss_fn = nn.MSELoss()
            return "continuous", action_dim, loss_fn

        raise pufferlib.APIUsageError(f"Unsupported action space for inverse dynamics: {action_space}")

    def _obs_tensor(self, observation):
        arr = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        return torch.from_numpy(arr).to(self.device)

    def encode(self, observation):
        """Return the latent representation φ(s)."""
        obs_tensor = self._obs_tensor(observation)
        with torch.no_grad():
            latent = self.encoder(obs_tensor)
        return latent.cpu().numpy().ravel().astype(np.float32, copy=False)

    def update(self, observation, action, next_observation):
        """One gradient step on the inverse-dynamics objective."""
        obs = self._obs_tensor(observation)
        next_obs = self._obs_tensor(next_observation)
        phi = self.encoder(obs)
        phi_next = self.encoder(next_obs)
        inv_input = torch.cat([phi, phi_next], dim=-1)
        predicted = self.inverse_model(inv_input)

        if self.action_type == "discrete":
            target = torch.as_tensor(action, dtype=torch.long, device=self.device).view(-1)
            loss = self.loss_fn(predicted, target)
        else:
            action_array = np.asarray(action, dtype=np.float32).reshape(1, -1)
            target = torch.from_numpy(action_array).to(self.device)
            loss = self.loss_fn(predicted, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        value = float(loss.item())
        self.last_loss = value
        return value


def make_inverse_dynamics_encoder(
    observation_space,
    action_space,
    latent_dim=128,
    hidden_dim=256,
    lr=1e-3,
    device=None,
):
    """Convenience helper to build an inverse-dynamics encoder from env spaces."""
    obs_shape = getattr(observation_space, "shape", None)
    if obs_shape is None:
        raise pufferlib.APIUsageError("Observation space must expose a shape attribute")

    return InverseDynamicsEncoder(
        observation_shape=obs_shape,
        action_space=action_space,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        device=device,
    )


def set_state_encoder(encoder):
    """Register a global encoder used by the default state_to_cell implementation."""
    global _STATE_ENCODER
    _STATE_ENCODER = encoder


def get_state_encoder():
    return _STATE_ENCODER


def make_vec_tracker(vecenv, cell_store=None, cell_encoder=None, on_cell=None,
    encoder_update_interval=None):
    if cell_store is None:
        cell_store = make_cell_store()
    if cell_encoder is None:
        cell_encoder = state_to_cell

    driver = getattr(vecenv, "driver_env", None)
    if driver is None:
        raise pufferlib.APIUsageError("Vector env must expose driver_env for Go-Explore tracking")

    agents_per_env = getattr(driver, "num_agents", None)
    if agents_per_env is None or agents_per_env < 1:
        raise pufferlib.APIUsageError("driver_env.num_agents must be >= 1")

    total_agents = getattr(vecenv, "agents_per_batch", getattr(vecenv, "num_agents", agents_per_env))
    if total_agents % agents_per_env != 0:
        raise pufferlib.APIUsageError("agents_per_batch must be divisible by driver_env.num_agents")

    num_envs = total_agents // agents_per_env

    tracker = {
        "vecenv": vecenv,
        "cell_store": cell_store,
        "encode_cell": cell_encoder,
        "on_cell": on_cell,
        "agents_per_env": agents_per_env,
        "num_envs": num_envs,
        "episode_returns": np.zeros(num_envs, dtype=np.float64),
        "step_counts": np.zeros(num_envs, dtype=np.int64),
        "trajectories": [[] for _ in range(num_envs)],
        "episode_seeds": [None for _ in range(num_envs)],
        "last_cell_key": [None for _ in range(num_envs)],
        "prev_observations": [None for _ in range(num_envs)],
        "encoder_update_interval": encoder_update_interval,
        "encoder_update_counter": 0,
    }

    return tracker


def _register_cells(tracker, observations):
    num_envs = tracker["num_envs"]
    agents_per_env = tracker["agents_per_env"]
    reshaped = reshape_per_env(observations, num_envs, agents_per_env)

    for env_idx in range(num_envs):
        encoded = tracker["encode_cell"](reshaped[env_idx])
        if isinstance(encoded, tuple) and len(encoded) == 2:
            cell_key, cell_observation = encoded
        else:
            cell_key = encoded
            cell_observation = np.array(reshaped[env_idx], copy=True)

        if cell_key is None:
            continue
        if tracker["last_cell_key"][env_idx] == cell_key:
            continue

        trajectory_steps = [
            np.array(step, copy=True) for step in tracker["trajectories"][env_idx]
        ]
        seed = tracker["episode_seeds"][env_idx]
        if seed is None:
            seed = _get_env_seed(tracker, env_idx)

        env_state = make_env_state(
            seed=seed,
            extras={
                "env_index": env_idx,
                "steps": int(tracker["step_counts"][env_idx]),
            },
        )
        cell = make_cell(
            cell_key,
            np.array(cell_observation, copy=True),
            env_state=env_state,
            reward=float(tracker["episode_returns"][env_idx]),
            trajectory=tuple(trajectory_steps),
            metadata={"env_index": env_idx},
        )
        created, stored = add_cell(tracker["cell_store"], cell)
        tracker["last_cell_key"][env_idx] = cell_key
        if created and tracker["on_cell"] is not None:
            tracker["on_cell"](stored)


def _cache_observations(tracker, observations):
    reshaped = reshape_per_env(observations, tracker["num_envs"], tracker["agents_per_env"])
    tracker["prev_observations"] = [np.array(obs, copy=True) for obs in reshaped]


def _maybe_update_encoder(tracker, actions, next_observations):
    encoder = get_state_encoder()
    interval = tracker.get("encoder_update_interval")
    if encoder is None or interval is None or interval <= 0:
        _cache_observations(tracker, next_observations)
        return

    tracker["encoder_update_counter"] += 1
    if tracker["encoder_update_counter"] % interval != 0:
        _cache_observations(tracker, next_observations)
        return

    reshaped_actions = reshape_per_env(actions, tracker["num_envs"], tracker["agents_per_env"])
    reshaped_next = reshape_per_env(next_observations, tracker["num_envs"], tracker["agents_per_env"])
    for env_idx in range(tracker["num_envs"]):
        prev_obs = tracker["prev_observations"][env_idx]
        if prev_obs is None:
            continue
        encoder.update(prev_obs, reshaped_actions[env_idx], reshaped_next[env_idx])

    tracker["prev_observations"] = [np.array(obs, copy=True) for obs in reshaped_next]


def tracker_reset(tracker, seed=None):
    observations, infos = tracker["vecenv"].reset(seed=seed)
    seeds = _resolve_seeds(tracker["num_envs"], seed)
    for env_idx, env_seed in enumerate(seeds):
        actual_seed = env_seed
        if actual_seed is None:
            actual_seed = _get_env_seed(tracker, env_idx)
        _reset_episode(tracker, env_idx, actual_seed)

    _register_cells(tracker, observations)
    _cache_observations(tracker, observations)
    return observations, infos


def tracker_step(tracker, actions):
    actions = np.asarray(actions)
    if actions.shape[0] != tracker["num_envs"] * tracker["agents_per_env"]:
        raise pufferlib.APIUsageError("Go-Explore tracker expects per-agent actions matching the vec env layout")

    reshaped_actions = reshape_per_env(actions, tracker["num_envs"], tracker["agents_per_env"])
    for env_idx in range(tracker["num_envs"]):
        tracker["trajectories"][env_idx].append(np.array(reshaped_actions[env_idx], copy=True))

    observations, rewards, terminals, truncations, infos = tracker["vecenv"].step(actions)
    reshaped_rewards = reshape_per_env(rewards, tracker["num_envs"], tracker["agents_per_env"])
    tracker["episode_returns"] += reshaped_rewards.sum(axis=1)
    tracker["step_counts"] += 1

    dones = reshape_per_env(np.logical_or(terminals, truncations), tracker["num_envs"], tracker["agents_per_env"])
    done_flags = dones.any(axis=1)
    for env_idx, env_done in enumerate(done_flags):
        if env_done:
            new_seed = _get_env_seed(tracker, env_idx)
            _reset_episode(tracker, env_idx, seed=new_seed)

    _register_cells(tracker, observations)
    _maybe_update_encoder(tracker, actions, observations)

    return observations, rewards, terminals, truncations, infos


def _sample_cells(cell_store, sampler_fn):
    if isinstance(cell_store, CellStore):
        return cell_store.sample(sampler_fn)
    return sampler_fn(cell_store["cells"])


def _cell_count(cell_store):
    if isinstance(cell_store, CellStore):
        return len(cell_store)
    return len(cell_store["cells"])


def _sample_with_weights(cells, weight_fn):
    if not cells:
        return None

    keys = list(cells.keys())
    entries = []
    weights = []
    for key in keys:
        cell = cells[key]
        entries.append((key, cell))
        weights.append(weight_fn(cell))

    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        idx = np.random.randint(len(entries))
    else:
        weights /= total
        idx = np.random.choice(len(entries), p=weights)

    key, cell = entries[idx]
    cell["visits"] += 1
    cell["selected"] += 1
    cells[key] = cell
    return cell


def sample_uniform_cell(cell_store):
    def _sampler(cells):
        return _sample_with_weights(cells, lambda cell: 1.0 / (1.0 + cell["visits"]))

    return _sample_cells(cell_store, _sampler)


def sample_weighted_cell(cell_store, reward_eps=1e-3):
    def _sampler(cells):
        def _weight(cell):
            w = cell["cumulative_reward"] / (1 + cell["visits"])
            if not np.isfinite(w) or w <= 0:
                return reward_eps
            return w

        return _sample_with_weights(cells, _weight)

    return _sample_cells(cell_store, _sampler)


def random_action_sampler(vecenv):
    space = vecenv.single_action_space
    samples = [space.sample() for _ in range(vecenv.num_agents)]
    return np.asarray(samples)


def return_to_cell(tracker, cell):
    env_state = cell.get("env_state") or {}
    seed = env_state.get("seed")
    if seed is None:
        seed = 0
    tracker_reset(tracker, seed=seed)

    for action in cell.get("trajectory", []):
        actions = np.asarray(action)
        tracker_step(tracker, actions)

    observations = tracker["vecenv"].observations
    key, _ = tracker["encode_cell"](observations)
    return key == cell["key"]


def explore_from_cell(tracker, steps, action_sampler=None):
    if action_sampler is None:
        action_sampler = random_action_sampler

    for _ in range(steps):
        actions = action_sampler(tracker["vecenv"])
        tracker_step(tracker, actions)


def go_explore_loop(return_env, cell_store, iterations=1, explore_steps=100,
        sample_cell_fn=None, action_sampler=None, encoder_update_interval=None):
    tracker = make_vec_tracker(
        return_env,
        cell_store=cell_store,
        encoder_update_interval=encoder_update_interval,
    )
    tracker_reset(tracker, seed=0)

    results = []
    sampler = sample_cell_fn or sample_weighted_cell
    for i in range(iterations):
        cell = sampler(cell_store)
        if cell is None:
            break

        env_seed = (cell.get("env_state") or {}).get("seed")
        if env_seed is None:
            results.append({"iteration": i, "cells": _cell_count(cell_store), "status": "no_seed"})
            continue

        matched = return_to_cell(tracker, cell)
        if not matched:
            results.append({"iteration": i, "cells": _cell_count(cell_store), "status": "return_failed"})
            continue

        explore_from_cell(tracker, explore_steps, action_sampler)
        results.append({"iteration": i, "cells": _cell_count(cell_store), "status": "ok"})

    return results


def _has_termination(terminals, truncations):
    terminals = np.asarray(terminals)
    truncations = np.asarray(truncations)
    return bool(terminals.any() or truncations.any())


def restore_cell(env, cell, action_transform=None, stop_on_done=True):
    env_state = cell.get("env_state") or {}
    seed = env_state.get("seed")

    if seed is None:
        observations, infos = env.reset()
    else:
        observations, infos = env.reset(seed=seed)

    if not cell["trajectory"]:
        return observations, infos

    for action in cell["trajectory"]:
        playback = np.array(action, copy=True)
        if action_transform is not None:
            playback = action_transform(playback)

        results = env.step(playback)
        if len(results) != 5:
            raise pufferlib.APIUsageError("restore_cell expects env.step to return (obs, rewards, terminals, truncations, infos)")

        observations, rewards, terminals, truncations, infos = results
        if stop_on_done and _has_termination(terminals, truncations):
            break

    return observations, infos


def restore_cell_from_tracker(tracker, cell, env_idx=None, **kwargs):
    vecenv = tracker["vecenv"]

    env = None
    if hasattr(vecenv, "envs"):
        envs = vecenv.envs
        if env_idx is None:
            env_idx = cell.get("metadata", {}).get("env_index", 0)
        if env_idx >= len(envs):
            raise pufferlib.APIUsageError(f"env_idx {env_idx} out of range for tracker restore")
        env = envs[env_idx]
    elif isinstance(vecenv, pufferlib.PufferEnv):
        env = vecenv
    else:
        raise pufferlib.APIUsageError("restore_cell_from_tracker supports Serial vecenvs or native PufferEnv instances")

    return restore_cell(env, cell, **kwargs)


def state_to_cell(observation):
    """Map observations to (cell_key, latent_representation) using the active encoder."""
    flat_obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    if _KEY_DECIMALS is not None:
        flat_obs = np.round(flat_obs, decimals=_KEY_DECIMALS)

    key = hashlib.sha1(flat_obs.tobytes()).hexdigest()
    encoder = get_state_encoder()
    if encoder is None:
        return key, flat_obs.copy()

    latent = encoder.encode(observation)
    return key, latent.copy()
