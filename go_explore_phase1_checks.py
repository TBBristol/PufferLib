import multiprocessing as mp
import time

import numpy as np

from pufferlib import go_explore as ge
from pufferlib.go_explore_phase1 import ReturnTask, _discovery_loop, _return_loop, run_phase1
from pufferlib.ge_reset_wrapper import GoExploreResetWrapper
from pufferlib.ocean.pong.pong import Pong


def _make_pong_env(num_envs, seed):
    return GoExploreResetWrapper(Pong(num_envs=num_envs, seed=seed))


def discovery_env_small():
    return _make_pong_env(num_envs=1, seed=1337)


def return_env_small():
    return _make_pong_env(num_envs=1, seed=2001)


def discovery_env_large():
    return _make_pong_env(num_envs=1024, seed=9001)


def _wait(seconds=1.0):
    time.sleep(seconds)


def measure_discovery_rate(env_fn, label, duration=3.0, steps_per_batch=4):
    cell_store = ge.make_shared_cell_store()
    stop_event = mp.Event()
    proc = mp.Process(
        target=_discovery_loop,
        args=(env_fn, cell_store, stop_event),
        kwargs={"steps_per_batch": steps_per_batch},
    )
    proc.start()

    start = time.time()
    last_time = start
    last_count = 0
    try:
        while time.time() - start < duration:
            _wait(1.0)
            current = len(cell_store)
            now = time.time()
            delta_cells = current - last_count
            delta_time = max(now - last_time, 1e-9)
            rate = delta_cells / delta_time
            print(f"[{label}] cells={current} rate={rate:.1f}/s")
            last_time = now
            last_count = current
    finally:
        stop_event.set()
        proc.join(timeout=5.0)

    return cell_store


def run_return_worker(cell_store, env_fn, explore_steps=4):
    keys = list(cell_store.keys())
    if not keys:
        print("return worker test skipped: empty cell store")
        return

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    stop_event = mp.Event()
    proc = mp.Process(
        target=_return_loop,
        args=(env_fn, cell_store, task_queue, result_queue, stop_event),
    )
    proc.start()
    try:
        cell_key = keys[0]
        task_queue.put(ReturnTask(cell_key=cell_key, explore_steps=explore_steps))
        result = result_queue.get(timeout=10.0)
        print("return worker result:", result)
    finally:
        stop_event.set()
        task_queue.put(None)
        proc.join(timeout=5.0)


def verify_return_to_cell(cell_store, env_fn):
    keys = list(cell_store.keys())
    if not keys:
        print("return verification skipped: empty cell store")
        return

    cell_key = keys[0]
    cell = cell_store.get(cell_key)
    env = env_fn()
    tracker = ge.make_vec_tracker(env, cell_store=ge.make_cell_store())
    ge.tracker_reset(tracker, seed=0)
    matched = ge.return_to_cell(tracker, cell)
    env.close()
    print(f"direct return check for {cell_key[:8]}:", "matched" if matched else "failed")


def run_missing_cell_check(cell_store, env_fn):
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    stop_event = mp.Event()
    proc = mp.Process(
        target=_return_loop,
        args=(env_fn, cell_store, task_queue, result_queue, stop_event),
    )
    proc.start()
    try:
        task_queue.put(ReturnTask(cell_key="deadbeef", explore_steps=1))
        result = result_queue.get(timeout=5.0)
        print("missing-cell result:", result)
    finally:
        stop_event.set()
        task_queue.put(None)
        proc.join(timeout=5.0)


def run_return_failure_check(cell_store, env_fn):
    keys = list(cell_store.keys())
    if not keys:
        print("return failure check skipped: empty cell store")
        return

    broken = cell_store.get(keys[0]).copy()
    broken["env_state"] = {}
    cell_store.add_cell(broken)

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    stop_event = mp.Event()
    proc = mp.Process(
        target=_return_loop,
        args=(env_fn, cell_store, task_queue, result_queue, stop_event),
    )
    proc.start()
    try:
        task_queue.put(ReturnTask(cell_key=broken["key"], explore_steps=1))
        result = result_queue.get(timeout=5.0)
        print("return-failure result:", result)
    finally:
        stop_event.set()
        task_queue.put(None)
        proc.join(timeout=5.0)


def run_controller_warmup_and_shutdown(discovery_fn, return_fn):
    logs = []

    def log_fn(msg):
        logs.append(msg)
        print("controller log:", msg)

    start = time.time()
    run_phase1(
        discovery_fn,
        return_fn,
        total_iterations=1,
        discovery_steps_per_iter=4,
        explore_steps=1,
        num_return_workers=1,
        log_fn=log_fn,
    )
    duration = time.time() - start
    print(f"controller run duration: {duration:.2f}s")
    print("controller logs captured:", len(logs))


def main():
    print("=== Discovery rate (small env) ===")
    small_store = measure_discovery_rate(discovery_env_small, "small", duration=3.0, steps_per_batch=8)
    print("Total cells stored (small):", len(small_store))

    print("=== Return worker single env ===")
    run_return_worker(small_store, return_env_small)
    verify_return_to_cell(small_store, return_env_small)
    run_missing_cell_check(small_store, return_env_small)
    run_return_failure_check(small_store, return_env_small)

    print("=== Discovery rate (1024 envs) ===")
    large_store = measure_discovery_rate(discovery_env_large, "large-1024", duration=3.0, steps_per_batch=2)
    print("Total cells stored (1024 envs):", len(large_store))

    print("=== Return worker after large discovery ===")
    run_return_worker(large_store, discovery_env_large)
    verify_return_to_cell(large_store, discovery_env_large)

    print("=== Controller warmup / iteration / shutdown ===")
    run_controller_warmup_and_shutdown(discovery_env_small, return_env_small)


if __name__ == "__main__":
    main()
