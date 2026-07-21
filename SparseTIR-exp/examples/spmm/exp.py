import os
import pathlib
import shlex
import subprocess
from tqdm import tqdm
from datetime import datetime
import sys
import argparse

script_path = pathlib.Path(__file__).resolve().parent
repo_root = script_path.parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mat-list", type=str, default=str(script_path / "mat_list.txt")
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(repo_root / "dataset"),
        help="Dataset root used to resolve relative paths in mat_list.txt.",
    )
    parser.add_argument(
        "--prog-list", type=str, default=str(script_path / "prog_list.txt")
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate matrix and program lists without running benchmarks.",
    )
    args = parser.parse_args()

    mat_list_path = pathlib.Path(args.mat_list).expanduser().resolve()
    prog_list_path = pathlib.Path(args.prog_list).expanduser().resolve()
    dataset_dir = pathlib.Path(args.dataset_dir).expanduser().resolve()

    with open(mat_list_path, "r") as f:
        mat_entries = [
            line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")
        ]

    mats = []
    for entry in mat_entries:
        mat_path = pathlib.Path(entry).expanduser()
        if not mat_path.is_absolute():
            mat_path = dataset_dir / mat_path
        mats.append(mat_path.resolve())

    missing_mats = [str(mat) for mat in mats if not mat.is_file()]
    if missing_mats:
        missing = "\n  ".join(missing_mats)
        raise FileNotFoundError(f"Matrix files not found:\n  {missing}")

    with open(prog_list_path, "r") as f:
        progs = [
            line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")
        ]

    print(f"Dataset directory: {dataset_dir}")
    print(f"Loaded {len(mats)} matrix paths from {mat_list_path}")
    print(progs)

    if args.validate_only:
        print("Input validation completed successfully.")
        raise SystemExit(0)

    # log_path = script_path / f"all_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    log_path = script_path / "sparsetir.log"
    with open(log_path, "w") as f:
        # Create progress bars outside the loops
        mat_bar = tqdm(mats, desc="Matrix", position=1, leave=False)
        for mat in mat_bar:
            prog_bar = tqdm(progs, desc="Program", position=2, leave=False)
            for prog in prog_bar:
                # Get current date
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                # Run the main command
                command = shlex.split(prog) + ["--mtx-file", str(mat)]
                shell_cmd = shlex.join(command)
                f.write(shell_cmd + "\n")
                f.flush()  # Ensure command is written to file immediately

                process = subprocess.Popen(
                    command,
                    cwd=script_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,  # Enable text mode
                )

                # Stream output in real-time
                while True:
                    output = process.stdout.readline()
                    error = process.stderr.readline()

                    if output == "" and error == "" and process.poll() is not None:
                        break

                    if output:
                        print(output.strip())  # Print to console
                        f.write(output)  # Write to file
                        f.flush()  # Ensure immediate file write
                    if error:
                        print(error.strip(), file=sys.stderr)  # Print to console
                        f.write(error)  # Write to file
                        f.flush()  # Ensure immediate file write

                f.write("=" * 100 + "\n")
                f.flush()  # Ensure separator is written to file

                # Update all progress bars
                prog_bar.refresh()
                mat_bar.refresh()
