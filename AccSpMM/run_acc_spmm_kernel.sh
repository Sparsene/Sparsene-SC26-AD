#!/bin/bash

ACCSPMM_DIR=$(pwd)
REPO_ROOT=$(cd "$ACCSPMM_DIR/.." && pwd)

matrix_list_file=$REPO_ROOT/dataset/filtered_mtx.txt
matrix_list_dir=$(cd "$(dirname "$matrix_list_file")" && pwd)


# feature_dims=(128 256 512)
feature_dims=(512)

while IFS= read -r matrix_path || [[ -n "$matrix_path" ]]; do
    [[ -z "$matrix_path" ]] && continue
    if [[ "$matrix_path" != /* ]]; then
        matrix_path="$matrix_list_dir/$matrix_path"
    fi

    for dim in "${feature_dims[@]}"; do
        echo "Running: ./mma $matrix_path $dim"
        "$ACCSPMM_DIR/mma" "$matrix_path" "$dim"
    done
done < "$matrix_list_file"
