
import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, capacity, sample_obs, sample_action, sample_goal, latent_dim):
        self.capacity = int(capacity)
        self.transitions_stored = 0
        self.full = False
        # Flat ring buffer: each row is one transition aligned across all arrays.

        obs = np.asarray(sample_obs)
        act = np.asarray(sample_action)
        goal = np.asarray(sample_goal)

        self.observations = np.empty((capacity, *obs.shape), dtype=obs.dtype)
        self.next_observations = np.empty((capacity, *obs.shape), dtype=obs.dtype)
        self.goals = np.empty((capacity, *goal.shape), dtype=goal.dtype)

        self.actions = np.empty((capacity, *act.shape), dtype=act.dtype)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.terminals = np.empty((capacity,), dtype=np.bool_)
        self.truncations = np.empty((capacity,), dtype=np.bool_)
        self.episode_id = np.empty((capacity,), dtype=np.int32)
        self.timestep = np.empty((capacity,), dtype=np.int32)
        self.latents = np.empty((capacity, latent_dim), dtype=np.float32)
        self.goal_latents = np.empty((capacity, latent_dim), dtype=np.float32)
        self.sorted_density = np.array([], dtype=np.int64)

    def sample(self, batch_size, rng):
        valid = self.capacity if self.full else self.transitions_stored
        if valid == 0:
            raise ValueError("ReplayBuffer.sample called with empty buffer")

        return rng.integers(0, valid, size=batch_size, dtype=np.int64)

    def add_batch(self, obs, next_obs, actions, rewards, dones, truncs, goals,
                  episode_ids, timesteps, latents=None, goal_latents=None):
        n = dones.shape[0]
        start = self.transitions_stored % self.capacity
        end = start + n

        if end <= self.capacity:
            sl = slice(start, end)
            self.observations[sl] = obs
            self.next_observations[sl] = next_obs
            self.actions[sl] = actions
            self.rewards[sl] = rewards
            self.terminals[sl] = dones
            self.truncations[sl] = truncs
            self.goals[sl] = goals
            self.episode_id[sl] = episode_ids
            self.timestep[sl] = timesteps
            if latents is not None:
                self.latents[sl] = latents
            if goal_latents is not None:
                self.goal_latents[sl] = goal_latents
        else:
            first = self.capacity - start
            second = n - first
            sl1 = slice(start, self.capacity)
            sl2 = slice(0, second)
            self.observations[sl1] = obs[:first]
            self.next_observations[sl1] = next_obs[:first]
            self.actions[sl1] = actions[:first]
            self.rewards[sl1] = rewards[:first]
            self.terminals[sl1] = dones[:first]
            self.truncations[sl1] = truncs[:first]
            self.goals[sl1] = goals[:first]
            self.episode_id[sl1] = episode_ids[:first]
            self.timestep[sl1] = timesteps[:first]
            if latents is not None:
                self.latents[sl1] = latents[:first]
            if goal_latents is not None:
                self.goal_latents[sl1] = goal_latents[:first]

            self.observations[sl2] = obs[first:]
            self.next_observations[sl2] = next_obs[first:]
            self.actions[sl2] = actions[first:]
            self.rewards[sl2] = rewards[first:]
            self.terminals[sl2] = dones[first:]
            self.truncations[sl2] = truncs[first:]
            self.goals[sl2] = goals[first:]
            self.episode_id[sl2] = episode_ids[first:]
            self.timestep[sl2] = timesteps[first:]
            if latents is not None:
                self.latents[sl2] = latents[first:]
            if goal_latents is not None:
                self.goal_latents[sl2] = goal_latents[first:]

        self.transitions_stored += n
        if self.transitions_stored >= self.capacity:
            self.full = True

    def recompute_sorted_density(self, device, k=1000, chunk=256, max_points=50000, rng=None):
        valid = self.capacity if self.full else self.transitions_stored
        if rng is None:
            rng = np.random.default_rng()
        if valid > max_points:
            indices = rng.choice(valid, size=max_points, replace=False)
        else:
            indices = np.arange(valid)
        embeddings = torch.as_tensor(self.latents[indices], device=device)
        if k > valid:
            k = valid

        density = np.empty(len(indices), dtype=np.float32)
        for start in range(0, len(indices), chunk):
            end = min(len(indices), start + chunk)
            cdist = torch.cdist(embeddings[start:end], embeddings)
            dist_to_k = cdist.topk(k, largest=False).values[:, -1]
            density[start:end] = (-dist_to_k).detach().cpu().numpy()

        self.density = density
        self.sorted_density = indices[np.argsort(density)]

    def recompute_latents(self, encoder, device, chunk=256):
        valid = self.capacity if self.full else self.transitions_stored
        for start in range(0, valid, chunk):
            end = min(valid, start + chunk)
            obs = torch.as_tensor(self.next_observations[start:end], device=device).float()
            goals = torch.as_tensor(self.goals[start:end], device=device).float()
            self.latents[start:end] = encoder(obs).detach().cpu().numpy()
            self.goal_latents[start:end] = encoder(goals).detach().cpu().numpy()

    def sample_pruned_trajectory(self, p, distance_threshold, rng, lighten_dist_coef=1.0):
        valid = self.capacity if self.full else self.transitions_stored

        # 1) Sample a low-density state via geometric rank on sorted density.
        rank = int(rng.geometric(p)) - 1
        if rank >= len(self.sorted_density):
            rank = len(self.sorted_density) - 1
        goal_idx = int(self.sorted_density[rank])
        if goal_idx >= valid:
            goal_idx = valid - 1

        # 2) Identify its episode and step.
        episode_id = self.episode_id[goal_idx]
        goal_step = self.timestep[goal_idx]

        # 3) Collect episode transitions up to the goal and sort by timestep.
        mask = self.episode_id[:valid] == episode_id
        indices = np.nonzero(mask)[0]
        indices = indices[self.timestep[indices] <= goal_step]
        order = np.argsort(self.timestep[indices])
        indices = indices[order]

        # 4) Prune subgoals in latent space based on distance threshold.
        latents = self.latents[indices]
        keep = [0]
        last_latent = latents[0]
        for i in range(1, indices.size - 1):
            dist = np.linalg.norm(latents[i] - last_latent)
            if dist >= distance_threshold * lighten_dist_coef:
                keep.append(i)
                last_latent = latents[i]
        if indices.size > 1:
            keep.append(indices.size - 1)
        return indices[np.array(keep, dtype=np.int64)]
