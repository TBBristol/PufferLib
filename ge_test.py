import numpy as np

from pufferlib.ocean.pong.pong import Pong
from pufferlib import go_explore as ge
from pufferlib.ge_reset_wrapper import GoExploreResetWrapper


def run_test(num_envs=4, steps=100000, seed=0):
    env = GoExploreResetWrapper(Pong(num_envs=num_envs, seed=seed))
    encoder = ge.make_inverse_dynamics_encoder(
        env.observation_space,
        env.action_space,
        latent_dim=64,
        hidden_dim=128,
    )
    encoder.last_loss = None
    ge.set_state_encoder(encoder)
    tracker = ge.make_vec_tracker(env, encoder_update_interval=100000)

    ge.tracker_reset(tracker, seed=seed)
    logged_updates = 0
    for _ in range(steps):
        actions = np.array(
            [env.single_action_space.sample() for _ in range(env.num_agents)],
            dtype=np.int32,
        )
        ge.tracker_step(tracker, actions)
        interval = tracker["encoder_update_interval"]
        counter = tracker["encoder_update_counter"]
        if (interval and interval > 0
                and counter % interval == 0
                and encoder.last_loss is not None
                and logged_updates < 5):
            print(f"encoder loss: {encoder.last_loss:.6f}")
            logged_updates += 1

    cell_store = tracker["cell_store"]
    snapshot = cell_store.copy_cells()
    print("cells stored:", len(snapshot))
    if snapshot:
        keys = list(snapshot.keys())
        for key in keys[:5]:
            cell = snapshot[key]
            restored_obs, _ = ge.restore_cell(tracker["vecenv"], cell)
            restored_key, _ = ge.state_to_cell(restored_obs)
            print(
                "cell",
                key[:8],
                "reward",
                cell["cumulative_reward"],
                "len",
                cell["trajectory_length"],
                "matches",
                restored_key == key,
            )

    return_env = GoExploreResetWrapper(Pong(num_envs=num_envs, seed=seed + 1))
    stats = ge.go_explore_loop(
        return_env,
        cell_store,
        iterations=2,
        explore_steps=10,
        encoder_update_interval=1000,
    )
    print("go-explore iterations:", len(stats))
    for entry in stats:
        print("  iteration", entry["iteration"], entry["status"])
    return_env.close()

    env.close()


if __name__ == "__main__":
    run_test()
