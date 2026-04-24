#!/usr/bin/env python3
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def convert_simple_npz_to_baseline_npz(input_npz, output_dir):
    folder_name = os.path.splitext(os.path.basename(input_npz))[0]
    output_path = os.path.join(output_dir, folder_name + '.npz')

    if os.path.exists(output_path):
        return f"{folder_name} skipped (already exists)"

    data = np.load(input_npz)
    src_li = data['src_li']
    dst_li = data['dst_li']
    num_nodes = int(data['num_nodes'])
    num_edges = len(src_li)

    num_nodes_src = num_nodes
    num_nodes_dst = num_nodes

    edge_index = np.stack([src_li, dst_li])
    adj = sp.coo_matrix((np.ones(num_edges), edge_index), shape=(num_nodes_src, num_nodes_dst))
    coo_mat = coo_matrix(adj)

    os.makedirs(output_dir, exist_ok=True)
    np.savez(output_path,
             num_nodes_src=num_nodes_src,
             num_nodes_dst=num_nodes_dst,
             num_edges=num_edges,
             src_li=coo_mat.row,
             dst_li=coo_mat.col,
             num_nodes=num_nodes)  
    return f"{folder_name} converted to baseline npz"

def convert_mtx_to_baseline_npz(mtx_file, output_dir):
    folder_name = os.path.splitext(os.path.basename(mtx_file))[0]
    output_path = os.path.join(output_dir, folder_name + '.npz')

    if os.path.exists(output_path):
        return f"{folder_name} skipped (already exists)"

    src_li1 = []
    dst_li1 = []
    with open(mtx_file, 'r') as file:
        for line in file:
            if line.startswith('%'):
                continue
            else:
                head = line.split()
                break

        for line in file:
            nums = line.strip().split()
            if len(nums) < 2:
                continue
            src_li1.append(int(nums[0]))
            dst_li1.append(int(nums[1]))

        num_nodes_src_ = int(head[0]) + 1
        num_nodes_dst_ = int(head[1]) + 1
        num_edges_ = int(head[2])
        edge_index = np.stack([src_li1, dst_li1])
        adj = sp.coo_matrix((np.ones(len(src_li1)), edge_index),
                            shape=(num_nodes_src_, num_nodes_dst_),
                            dtype=np.float32)
        coo_mat = coo_matrix(adj)

        os.makedirs(output_dir, exist_ok=True)
        np.savez(output_path,
                 num_nodes_src=num_nodes_src_,
                 num_nodes_dst=num_nodes_dst_,
                 num_edges=num_edges_,
                 src_li=coo_mat.row,
                 dst_li=coo_mat.col,
                 num_nodes=num_nodes_src_)  
    return f"{folder_name} converted from MTX to baseline npz"

def worker(file_path, output_dir, mode):
    if mode == 'npz' and file_path.endswith('.npz'):
        return convert_simple_npz_to_baseline_npz(file_path, output_dir)
    elif mode == 'mtx' and file_path.endswith('.mtx'):
        return convert_mtx_to_baseline_npz(file_path, output_dir)
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert simple npz or mtx to baseline npz")
    parser.add_argument('--input_dir', type=str, required=True, help="Folder containing simple npz or mtx files")
    parser.add_argument('--output_dir', type=str, required=True, help="Folder to save baseline npz")
    parser.add_argument('--mode', type=str, choices=['npz', 'mtx'], default='npz', help="Input file type: npz or mtx")
    parser.add_argument('--threads', type=int, default=4, help="Number of threads for parallel conversion")
    args = parser.parse_args()

    files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)]
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(worker, f, args.output_dir, args.mode) for f in files]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting files"):
            result = fut.result()
            if result:
                tqdm.write(result)
