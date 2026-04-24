import torch
import sys
import csv
import pandas as pd
import time
import os
import subprocess
from tqdm import tqdm
 

if __name__ == "__main__":
    repo_root = sys.argv[2] if len(sys.argv) > 2 else "/workspace/Sparsene-AD-repo"
    data_filter = sys.argv[3] if len(sys.argv) > 3 else f"{repo_root}/dataset/data_filter.csv"

    gpu_device = torch.cuda.current_device()
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)

    dimN = int(sys.argv[1])
    print('dimN: ' + str(dimN))

    epoches = 1
    partsize_t = 32
    
    current_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

    data_path = f"{repo_root}/dataset/flashsparse_npz/"
    file_name = project_dir + f'/result/FlashSparse/spmm/spmm_tf32_{dimN}_new.csv'

    head = ['dataSet', 'num_nodes', 'num_edges', '16_1', '8_1', '8_1_balance', '8_1_map']
    
    #                
    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(head)

    #         dataset（      ）
    completed_datasets = set()
    # if os.path.exists(file_name):
    #     existing_df = pd.read_csv(file_name)
    #     completed_datasets = set(existing_df['dataSet'].astype(str).tolist())
    # else:
    #     completed_datasets = set()
    
    # with open(file_name, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(head)

    start_time = time.time()
    df = pd.read_csv(data_filter)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"spmm_tf32_{dimN}"):
        dataset, num_nodes, num_edges = row.iloc[0], str(row.iloc[1]), str(row.iloc[2])
        
        if str(dataset) in completed_datasets:
            print(f"[SKIP] {dataset} already exists in CSV, skipping...")
            continue
        # if dataset != "mip1":
        #     continue
        cmd = [
            "python", f"{repo_root}/FlashSparse/eva/kernel/spmm/spmm_tf32_run_one.py",
            dataset, num_nodes, num_edges,
            str(dimN), str(epoches), str(partsize_t),
            data_path, file_name
        ]
        try:
            #      300  （5   ）
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"[ERROR] {dataset} returned code {result.returncode}")
                print(result.stderr.strip())
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {dataset} did not finish in 5 minutes, skipping...")


    print('All is success')

    end_time = time.time()
    execution_time = end_time - start_time
    with open("execution_time.txt", "a") as file:
        file.write(f"TF32-{dimN}-{round(execution_time/60,2)} minutes\n")
