import os

def collect_mtx_files(root_dir, output_file):
    mtx_files = []
    #      
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".mtx"):
                full_path = os.path.join(dirpath, filename)
                mtx_files.append(full_path)

    #     txt   
    with open(output_file, "w", encoding="utf-8") as f:
        for file_path in mtx_files:
            f.write(file_path + "\n")

    print(f"    {len(mtx_files)}   .mtx   ，    {output_file}")

if __name__ == "__main__":
    root_directory = "/workspace/selectedMM/"  #              
    output_txt = "/workspace/sparsene/examples/src_fp32/dtc/testbed/scripts/mtx_files.txt"
    collect_mtx_files(root_directory, output_txt)
