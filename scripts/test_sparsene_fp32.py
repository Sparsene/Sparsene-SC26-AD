#!/usr/bin/env python3
import argparse
import os
import subprocess
from tqdm import tqdm
import sys
from pathlib import Path

OLD_REPO_ROOT = "/workspace/Sparsene-AD-repo"


def normalize_dataset_path(raw_path: str, repo_root: Path) -> str:
    path = raw_path.strip()
    if not path:
        return ""
    if path.startswith(OLD_REPO_ROOT + "/"):
        relative = path[len(OLD_REPO_ROOT) + 1 :]
        return str(repo_root / relative)
    if os.path.isabs(path):
        return path
    return str(repo_root / path)


def load_exe_list(exe_list_file: Path, repo_root: Path):
    with exe_list_file.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    exe_list = []
    for exe in lines:
        if exe.startswith(OLD_REPO_ROOT + "/"):
            relative = exe[len(OLD_REPO_ROOT) + 1 :]
            exe_list.append(str(repo_root / relative))
        elif os.path.isabs(exe):
            exe_list.append(exe)
        else:
            exe_list.append(str(repo_root / "sparsene" / exe))
    return exe_list


def load_mtx_list(filtered_mtx_file: Path, repo_root: Path):
    with filtered_mtx_file.open("r") as f:
        return [normalize_dataset_path(line, repo_root) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, help="Feature dimension")
    parser.add_argument("repo_root", type=str, help="Repository root path")
    parser.add_argument(
        "--impl",
        choices=["sparsene", "cusparse"],
        default="sparsene",
        help="Backend to test (default: sparsene)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    exe_list_file = repo_root / "scripts" / f"{args.impl}_fp32_exe_list.txt"
    filtered_mtx_file = repo_root / "dataset" / "filtered_mtx.txt"
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dimN = args.N
    print("dimN: " + str(dimN))
    N = dimN
    log_file = results_dir / f"{args.impl}_fp32_N{N}.log"

    if not exe_list_file.exists():
        raise FileNotFoundError(f"Missing executable list file: {exe_list_file}")
    if not filtered_mtx_file.exists():
        raise FileNotFoundError(f"Missing matrix list file: {filtered_mtx_file}")

    exe_list = load_exe_list(exe_list_file, repo_root)
    mtx_list = load_mtx_list(filtered_mtx_file, repo_root)

    if not exe_list:
        raise RuntimeError(f"No executables found in {exe_list_file}")
    if not mtx_list:
        raise RuntimeError("No matrix entries found in dataset/filtered_mtx.txt")

    total_tasks = len(exe_list) * len(mtx_list)

    with log_file.open("w") as log:
        with tqdm(total=total_tasks, desc="Running tests", ncols=100) as pbar:
            for mtx in mtx_list:
                for exe in exe_list:
                    cmd = [exe, "-filename", mtx, "-N", str(N)]
                    try:
                        result = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                        log.write(f"===== Matrix: {mtx}, Executable: {exe} =====\n")
                        log.write(result.stdout)
                        log.write(result.stderr)
                        log.write("\n")
                    except Exception as e:
                        log.write(f"Error running {exe} with {mtx}: {e}\n")
                    pbar.update(1)


if __name__ == "__main__":
    main()
