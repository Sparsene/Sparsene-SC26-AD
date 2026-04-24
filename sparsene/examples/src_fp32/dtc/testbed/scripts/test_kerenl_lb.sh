#!/bin/bash

#        sparse_files.txt
input_file="/workspace/sparsene/example/dtc-spmm/scripts/filtered_mtx_files.txt.small"
input_file="/workspace/sparsene/examples/scripts/selected_mtx.2.txt"
input_file="/workspace/sparsene/examples/src_fp32/dtc/testbed/scripts/mtx_files2.txt"

#         
if [ ! -f "$input_file" ]; then
  echo "   $input_file    "
  exit 1
fi

#     txt           
while IFS= read -r line
do
  #       
  if [ -n "$line" ]; then
    echo "     $line ..."

    echo "Normal Binding"  
    timeout 300 ncu --clock-control none --set full -k regex:dtc* /workspace/sparsene/examples/src_fp32/dtc/testbed/build/demo-dtc-tf32 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000 -ncu 1 > /workspace/sparsene/examples/scripts/tmp.log 2>&1
    cat /workspace/sparsene/examples/scripts/tmp.log | grep -E "Duration |Waves Per SM|Active Warps [pP]er SM" 
    # /workspace/sparsene/examples/src_fp32/dtc/testbed/build/demo-dtc-tf32 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000 


    echo "Multi Binding"
    timeout 300 ncu --clock-control none --set full -k regex:dtc* /workspace/sparsene/examples/src_fp32/dtc/testbed/build/demo-dtc-multi-binding-tf32 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000 -ncu 1 > /workspace/sparsene/examples/scripts/tmp.log 2>&1
    cat /workspace/sparsene/examples/scripts/tmp.log | grep -E "Duration |Waves Per SM|Active Warps [pP]er SM" 
    # /workspace/sparsene/examples/src_fp32/dtc/testbed/build/demo-dtc-multi-binding-tf32 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000


    echo "Strict LB"
    timeout 300 ncu --clock-control none --set full -k regex:dtc* /workspace/sparsene/examples/src_fp32/dtc/testbed/build/demo-dtc-strict-lb-tf32 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000 -ncu 1 > /workspace/sparsene/examples/scripts/tmp.log 2>&1
    cat /workspace/sparsene/examples/scripts/tmp.log | grep -E "Duration |Waves Per SM|Active Warps [pP]er SM" 
    # /workspace/sparsene/examples/src_fp32/dtc/testbed/build/demo-dtc-strict-lb-tf32 -filename "$line" -N 512 -mtx_flag 1 -warmup 100 -repeat 1000
  fi
done < "$input_file"