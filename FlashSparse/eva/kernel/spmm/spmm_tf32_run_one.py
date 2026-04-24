# run_one_tf32.py
import sys
import csv
from fs_tf32 import test_fs

def fs_tf32_16_1(dataset, dimN, epoches, partsize, data_path, window, wide):       
    return test_fs.fs_tf32_16_1(dataset, epoches, dimN, partsize, data_path, window, wide)

def fs_tf32_8_1(dataset, dimN, epoches, partsize, data_path, window, wide):     
    return test_fs.fs_tf32_8_1(dataset, epoches, dimN, partsize, data_path, window, wide)

def fs_tf32_8_1_map(dataset, dimN, epoches, partsize, data_path, window, wide):     
    return test_fs.fs_tf32_8_1_map(dataset, epoches, dimN, partsize, data_path, window, wide)

def fs_tf32_8_1_balance(dataset, dimN, epoches, partsize, data_path, window, wide):     
    return test_fs.fs_tf32_8_1_balance(dataset, epoches, dimN, partsize, data_path, window, wide)

if __name__ == "__main__":
    dataset = sys.argv[1]
    num_nodes = sys.argv[2]
    num_edges = sys.argv[3]
    dimN = int(sys.argv[4])
    epoches = int(sys.argv[5])
    partsize_t = int(sys.argv[6])
    data_path = sys.argv[7]
    file_name = sys.argv[8]

    try:
        res_temp = [dataset, num_nodes, num_edges]

        res_temp.append(fs_tf32_16_1(dataset, dimN, epoches, partsize_t, data_path, 16, 4))
        res_temp.append(fs_tf32_8_1(dataset, dimN, epoches, partsize_t, data_path, 8, 4))
        res_temp.append(fs_tf32_8_1_balance(dataset, dimN, epoches, partsize_t, data_path, 8, 4))
        res_temp.append(fs_tf32_8_1_map(dataset, dimN, epoches, partsize_t, data_path, 8, 4))

        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res_temp)

        print(f"{dataset} success")
    except Exception as e:
        print(f"[ERROR] {dataset}: {e}")
        sys.exit(1)
