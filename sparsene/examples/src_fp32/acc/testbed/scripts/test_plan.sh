#!/bin/bash

#      
DIR="/workspace/sparsene/examples/src_fp32/acc/testbed/build"

#      acc-plan_xxx-tf32
for exe in "$DIR"/acc-plan_*"-tf32"; do
    if [[ -x "$exe" ]]; then
        echo "    : $exe"
        "$exe" -filename /workspace/selectedMM/cop20k_A.mtx -N 512
    else
        echo "  : $exe (       )"
    fi
done
