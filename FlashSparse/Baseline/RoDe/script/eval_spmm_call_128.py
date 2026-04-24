#!/usr/bin/env python3
import torch
import pandas as pd
import csv
import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", type=str, help="Repository root path")
    parser.add_argument("--data-filter", type=str, default=None, help="CSV file with dataset list")
    parser.add_argument("--mtx-dir", type=str, default=None, help="Directory containing .mtx files")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent.parent.parent
    data_filter = Path(args.data_filter).resolve() if args.data_filter else repo_root / "dataset" / "data_filter.csv"
    mtx_dir = Path(args.mtx_dir).resolve() if args.mtx_dir else repo_root / "dataset" / "selected_mtx"

    df = pd.read_csv(data_filter)
    file_name = project_dir / 'result' / 'Baseline' / 'spmm' / 'rode_spmm_f32_n128.csv'
    file_name.parent.mkdir(parents=True, exist_ok=True)
    head = ['dataSet','rows_','columns_','nonzeros_','sputnik','Sputnik_gflops','cusparse','cuSPARSE_gflops','rode','ours_gflops']

    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(head)
    count = 0

    start_time = time.time()
    binary = project_dir / 'Baseline' / 'RoDe' / 'build' / 'eval' / 'eval_spmm_f32_n128'
    for index, row in df.iterrows():
        count += 1
        data_set = str(row['dataSet'])
        mtx_file = mtx_dir / f'{data_set}.mtx'
        with open(file_name, 'a', newline='') as csvfile:
            csvfile.write(','.join(map(str, [data_set])))
        shell_command = f"{binary} {mtx_file} >> {file_name}"
        print(data_set)
        subprocess.run(shell_command, shell=True)

    end_time = time.time()
    execution_time = end_time - start_time

    dimN = 128
    with open("execution_time_base.txt", "a") as file:
        file.write("spmm-" + str(dimN) + "-" + str(round(execution_time/60,2)) + " minutes\n")


if __name__ == "__main__":
    main()