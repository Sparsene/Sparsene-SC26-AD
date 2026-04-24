import numpy as np
from scipy.io import mmread
import argparse

def convert_mtx_to_npz(input_file, output_file):
    # read mtx file, convert to COO format sparse matrix
    coo_matrix = mmread(input_file).tocoo()
    
    # Get the row and column arrays in COO format
    src_li = coo_matrix.row
    dst_li = coo_matrix.col
    
    # Get the edge length of the matrix (number of rows or columns)
    num_nodes = max(coo_matrix.shape)
    
    # Save as npz file
    np.savez(output_file, src_li=src_li, dst_li=dst_li, num_nodes=num_nodes)
    print(f"Converted {input_file} to {output_file}")

if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Convert .mtx to .npz")
    parser.add_argument("input_file", type=str, help="Input .mtx file path")
    parser.add_argument("output_file", type=str, help="Output .npz file path")
    args = parser.parse_args()
    print(f"converting {args.input_file} to {args.output_file}")
    # call the conversion function
    convert_mtx_to_npz(args.input_file, args.output_file)

