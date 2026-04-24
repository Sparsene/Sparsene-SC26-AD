import re
import json
import argparse
import os
from collections import defaultdict

def parse_log(log_file):
    fp32_data = defaultdict(lambda: defaultdict(dict))
    fp16_data = defaultdict(lambda: defaultdict(dict))

    current_matrix = None
    current_mode = None  # "naive" or "tc"
    feat_size = None

    with open(log_file, "r") as f:
        for line in f:
            #      
            m = re.search(r"--mtx-file .*?/([^/]+)\.mtx", line)
            if m:
                current_matrix = m.group(1)
                # print(current_matrix)
                if current_matrix == "DD": 
                    print(current_matrix)
                continue

            #     
            # if "bench_spmm_naive.py" in line:
            #     current_mode = "naive"
            # elif "bench_tc_spmm.py" in line:
            #     current_mode = "tc"

            #    feat_size
            m = re.search(r"feat_size\s*=\s*(\d+)", line)
            if m:
                feat_size = m.group(1)
                continue

            #    
            # if current_mode == "naive":
            if "tir naive time" in line:
                m = re.search(r"tir naive time:\s*([\d.]+)", line)
                print(current_mode, m)
                if m and current_matrix and feat_size:
                    fp32_data[current_matrix][feat_size]["sparsetir"] = float(m.group(1))
            # elif current_mode == "tc":
            elif "tc-spmm time" in line:
                m = re.search(r"tc-spmm time:\s*([\d.]+)", line)
                if m and current_matrix and feat_size:
                    fp16_data[current_matrix][feat_size]["sparsetir"] = float(m.group(1))

    return fp16_data, fp32_data


def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="   SparseTIR       JSON")
    parser.add_argument("log_file", help="        ")
    parser.add_argument("output_dir", help="       ")
    args = parser.parse_args()

    log_file = args.log_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    fp16_path = os.path.join(output_dir, "sparsetir_fp16.json")
    fp32_path = os.path.join(output_dir, "sparsetir_fp32.json")

    fp16_data, fp32_data = parse_log(log_file)

    save_json(fp16_data, fp16_path)
    save_json(fp32_data, fp32_path)

    print(fp32_data)

    print(f"    {fp16_path}   {fp32_path}")
