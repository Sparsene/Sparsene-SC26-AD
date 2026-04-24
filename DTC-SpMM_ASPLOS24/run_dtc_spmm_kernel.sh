#!/bin/bash

set -e

DTCSPMM_DIR=$(pwd)
REPO_ROOT=$(cd "$DTCSPMM_DIR/.." && pwd)
RUNNER="$REPO_ROOT/DTC-SpMM_ASPLOS24/scripts/DTCSpMM/run_DTC_SpMM_selected.py"

# Usage:
#   bash run_dtc_spmm_kernel.sh [mtx_list_file]
# If not provided, it uses dataset/filtered_mtx.txt under the repo root.
MTXFILE="${1:-$REPO_ROOT/dataset/filtered_mtx.txt}"

if [ ! -f "$RUNNER" ]; then
	echo "[ERROR] Missing runner: $RUNNER"
	exit 1
fi

if [ ! -f "$MTXFILE" ]; then
	echo "[ERROR] Missing matrix list file: $MTXFILE"
	echo "[NOTE] Pass a list file explicitly, e.g.:"
	echo "       bash run_dtc_spmm_kernel.sh /path/to/data_path_list.txt"
	exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
	entry=$(echo "$line" | xargs)
	if [ -z "$entry" ]; then
		continue
	fi

	# Keep the same behavior as the previous Python script:
	# mat_name = line.strip().split('/')[-1][:-4]
	base_name=$(basename "$entry")
	mat_name="${base_name%.*}"

	echo "Running DTC-SpMM for $mat_name:"
	python -u "$RUNNER" --dataset "$mat_name"
done < "$MTXFILE"

