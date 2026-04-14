#!/usr/bin/env python3
import argparse
import csv
import itertools
import math
import os
import pathlib
import re
import subprocess
import sys
from typing import Dict, Iterable, List


SUMMARY_RE = re.compile(
    r"trim_summary "
    r"terminated=(?P<terminated>\d+) "
    r"step=(?P<step>\d+) "
    r"score=(?P<score>[-+0-9.eE]+) "
    r"mean_abs_roll=(?P<mean_abs_roll>[-+0-9.eE]+) "
    r"mean_abs_pitch=(?P<mean_abs_pitch>[-+0-9.eE]+) "
    r"mean_abs_omega=(?P<mean_abs_omega>[-+0-9.eE]+) "
    r"max_abs_roll=(?P<max_abs_roll>[-+0-9.eE]+) "
    r"max_abs_pitch=(?P<max_abs_pitch>[-+0-9.eE]+) "
    r"max_abs_omega=(?P<max_abs_omega>[-+0-9.eE]+) "
    r"altitude_drift=(?P<altitude_drift>[-+0-9.eE]+) "
    r"speed_drift=(?P<speed_drift>[-+0-9.eE]+)"
)


def parse_floats(spec: str) -> List[float]:
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError(f"empty sweep spec: {spec!r}")
    return values


def iter_candidates(args: argparse.Namespace) -> Iterable[Dict[str, float]]:
    grid = itertools.product(
        parse_floats(args.pitch),
        parse_floats(args.speed),
        parse_floats(args.throttle_state),
        parse_floats(args.heading_action),
        parse_floats(args.altitude_action),
        parse_floats(args.throttle_action),
        parse_floats(args.k_pitch),
        parse_floats(args.k_pitch_rate),
        parse_floats(args.k_bank),
        parse_floats(args.k_roll_rate),
        parse_floats(args.k_altitude),
        parse_floats(args.max_pitch_cmd),
        parse_floats(args.k_beta),
        parse_floats(args.k_yaw_rate),
    )
    for (
        pitch,
        speed,
        throttle_state,
        heading_action,
        altitude_action,
        throttle_action,
        k_pitch,
        k_pitch_rate,
        k_bank,
        k_roll_rate,
        k_altitude,
        max_pitch_cmd,
        k_beta,
        k_yaw_rate,
    ) in grid:
        yield {
            "pitch": pitch,
            "speed": speed,
            "throttle_state": throttle_state,
            "heading_action": heading_action,
            "altitude_action": altitude_action,
            "throttle_action": throttle_action,
            "k_pitch": k_pitch,
            "k_pitch_rate": k_pitch_rate,
            "k_bank": k_bank,
            "k_roll_rate": k_roll_rate,
            "k_altitude": k_altitude,
            "max_pitch_cmd": max_pitch_cmd,
            "k_beta": k_beta,
            "k_yaw_rate": k_yaw_rate,
        }


def run_candidate(
    exe: pathlib.Path,
    out_dir: pathlib.Path,
    steps: int,
    pos: List[float],
    candidate: Dict[str, float],
    idx: int,
) -> Dict[str, float]:
    csv_path = out_dir / f"trim_{idx:04d}.csv"
    cmd = [
        str(exe),
        "headless",
        str(steps),
        "0",
        str(csv_path),
        "--action",
        str(candidate["heading_action"]),
        str(candidate["altitude_action"]),
        str(candidate["throttle_action"]),
        "--pos",
        str(pos[0]),
        str(pos[1]),
        str(pos[2]),
        "--pitch",
        str(candidate["pitch"]),
        "--speed",
        str(candidate["speed"]),
        "--throttle-state",
        str(candidate["throttle_state"]),
        "--k-altitude",
        str(candidate["k_altitude"]),
        "--max-pitch-cmd",
        str(candidate["max_pitch_cmd"]),
        "--k-pitch",
        str(candidate["k_pitch"]),
        "--k-pitch-rate",
        str(candidate["k_pitch_rate"]),
        "--k-bank",
        str(candidate["k_bank"]),
        "--k-roll-rate",
        str(candidate["k_roll_rate"]),
        "--k-beta",
        str(candidate["k_beta"]),
        "--k-yaw-rate",
        str(candidate["k_yaw_rate"]),
    ]
    env = dict(os.environ)
    env.setdefault("ASAN_OPTIONS", "detect_leaks=0")
    env.setdefault("LSAN_OPTIONS", "exitcode=0")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, env=env
    )
    stdout_path = out_dir / f"trim_{idx:04d}.stdout.txt"
    stderr_path = out_dir / f"trim_{idx:04d}.stderr.txt"
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)

    if proc.returncode != 0:
        result = {
            "terminated": 1,
            "step": 0,
            "score": 1e9 + float(idx),
            "mean_abs_roll": math.inf,
            "mean_abs_pitch": math.inf,
            "mean_abs_omega": math.inf,
            "max_abs_roll": math.inf,
            "max_abs_pitch": math.inf,
            "max_abs_omega": math.inf,
            "altitude_drift": math.inf,
            "speed_drift": math.inf,
            "csv_path": str(csv_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "returncode": proc.returncode,
            "crashed": 1,
        }
        result.update(candidate)
        return result

    match = SUMMARY_RE.search(proc.stdout)
    if not match:
        result = {
            "terminated": 1,
            "step": 0,
            "score": 1e8 + float(idx),
            "mean_abs_roll": math.inf,
            "mean_abs_pitch": math.inf,
            "mean_abs_omega": math.inf,
            "max_abs_roll": math.inf,
            "max_abs_pitch": math.inf,
            "max_abs_omega": math.inf,
            "altitude_drift": math.inf,
            "speed_drift": math.inf,
            "csv_path": str(csv_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "returncode": proc.returncode,
            "crashed": 1,
        }
        result.update(candidate)
        return result

    result = {k: float(v) for k, v in match.groupdict().items()}
    result["terminated"] = int(result["terminated"])
    result["step"] = int(result["step"])
    result["csv_path"] = str(csv_path)
    result["stdout_path"] = str(stdout_path)
    result["stderr_path"] = str(stderr_path)
    result["returncode"] = proc.returncode
    result["crashed"] = 0
    result.update(candidate)
    return result


def write_results_csv(path: pathlib.Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep BVR headless trim candidates and rank steady-flight scores."
    )
    parser.add_argument("--exe", default="./bvr", help="Path to built BVR executable")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--out-dir", default="/tmp/bvr_trim_search")
    parser.add_argument("--limit", type=int, default=64, help="Max candidates to run")
    parser.add_argument("--pos", default="0,0,75")
    parser.add_argument("--pitch", default="-0.08,-0.04,-0.02,0.0")
    parser.add_argument("--speed", default="35,45,55")
    parser.add_argument("--throttle-state", default="0.5,0.6,0.7")
    parser.add_argument("--heading-action", default="0")
    parser.add_argument("--altitude-action", default="-0.25,0,0.25")
    parser.add_argument("--throttle-action", default="0,0.5,1.0")
    parser.add_argument("--k-pitch", default="0.1,0.2,0.3")
    parser.add_argument("--k-pitch-rate", default="0.5,1.0,1.5")
    parser.add_argument("--k-bank", default="1.0,1.5,2.0")
    parser.add_argument("--k-roll-rate", default="0.4,0.8,1.2")
    parser.add_argument("--k-altitude", default="0.005,0.01,0.015")
    parser.add_argument("--max-pitch-cmd", default="0.03,0.06,0.1")
    parser.add_argument("--k-beta", default="0.2,0.5,1.0")
    parser.add_argument("--k-yaw-rate", default="0.2,0.5,1.0")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    exe = pathlib.Path(args.exe).resolve()
    if not exe.exists():
        print(f"missing executable: {exe}", file=sys.stderr)
        return 1

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = parse_floats(args.pos)
    if len(pos) != 3:
        raise ValueError("--pos must have three comma-separated floats")

    results: List[Dict[str, float]] = []
    for idx, candidate in enumerate(iter_candidates(args), start=1):
        if idx > args.limit:
            break
        result = run_candidate(exe, out_dir, args.steps, pos, candidate, idx)
        results.append(result)
        print(
            f"{idx:04d} score={result['score']:.4f} terminated={result['terminated']} "
            f"crashed={result['crashed']} pitch={result['pitch']:.3f} speed={result['speed']:.1f} "
            f"thr_state={result['throttle_state']:.2f} "
            f"act=({result['heading_action']:.2f},{result['altitude_action']:.2f},{result['throttle_action']:.2f}) "
            f"k_pitch={result['k_pitch']:.2f} k_pitch_rate={result['k_pitch_rate']:.2f} "
            f"max_pitch={result['max_pitch_cmd']:.3f}"
        )

    results.sort(
        key=lambda row: (row["crashed"], row["score"], row["terminated"], row["max_abs_omega"])
    )
    write_results_csv(out_dir / "results.csv", results)

    print("\nTop candidates:")
    for rank, row in enumerate(results[: args.top_k], start=1):
        print(
            f"{rank:02d} score={row['score']:.4f} term={row['terminated']} crashed={row['crashed']} "
            f"mean_roll={row['mean_abs_roll']:.3f} mean_pitch={row['mean_abs_pitch']:.3f} "
            f"mean_omega={row['mean_abs_omega']:.3f} alt_drift={row['altitude_drift']:.2f} "
            f"speed_drift={row['speed_drift']:.2f} csv={row['csv_path']}"
        )

    print(f"\nWrote {len(results)} candidates to {out_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
