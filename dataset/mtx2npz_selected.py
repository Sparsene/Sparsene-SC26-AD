from mtx2npz import convert_mtx_to_npz
import os
from pathlib import Path


if __name__ == "__main__":
	with open("/workspace/Sparsene-AD-repo/dataset/filtered_mtx.txt", "r") as f:
		for line in f:
			src = line.strip()
			dst = src.replace(".mtx", ".npz").split("/")[-1]
			output_dir = "/workspace/Sparsene-AD-repo/dataset/selected_npz"
			os.makedirs(output_dir, exist_ok=True)
			convert_mtx_to_npz(src, Path(output_dir) / dst)
