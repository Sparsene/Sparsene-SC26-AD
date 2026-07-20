import os
import json
import glob
import argparse
from collections import defaultdict

# Map filename keywords to the standardized method names used in the output.
METHOD_NAME_MAP = {
    "sparsene": "sparsene",
    "flashsparse": "FlashSparse",
    "dtc": "DTC-SPMM",
    "acc": "ACC-SPMM",
    "cusparse": "cusparse",
    "sputnik": "sputnik",
    "sparsetir": "SparseTIR"
}

# Only merge results for these output feature dimensions (N values).
TARGET_N_VALUES = {"128", "256", "512"}

def clean_dataset_name(name):
    """
    Normalize a dataset name by removing its .mtx suffix, commas, and
    surrounding whitespace.
    """
    return name.replace(".mtx", "").replace(",", "").strip()

def extract_fastest_time(method_dict):
    """
    Return the fastest numeric timing for one N configuration, or None when
    no valid numeric timing is available.
    """
    times = []
    if not isinstance(method_dict, dict):
        return None
        
    for key, value in method_dict.items():
        if isinstance(value, list):
            times.extend([v for v in value if isinstance(v, (int, float))])
        elif isinstance(value, (int, float)):
            times.append(value)
            
    if times:
        return min(times)
    return None

def get_standard_method_name(filename):
    """
    Infer the standardized baseline name from an input filename.
    """
    filename_lower = filename.lower()
    for key, standard_name in METHOD_NAME_MAP.items():
        if key in filename_lower:
            return standard_name
    return "Unknown"

def main():
    # 1. Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Merge JSON files for different baselines.")
    parser.add_argument(
        "-i", "--input_dir", 
        type=str, 
        default=".", 
        help="Directory containing the input JSON files (default: current directory)."
    )
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        default=".", 
        help="Directory where the merged JSON file will be written (default: current directory)."
    )
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}\n")

    # Create the output directory when it does not already exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_filename = "a100_fp32_merged2.json"
    target_filepath = os.path.join(output_dir, target_filename)
    
    merged_data = defaultdict(lambda: defaultdict(dict))
    
    # 2. Load an existing merged file so methods from earlier runs are kept.
    if os.path.exists(target_filepath):
        with open(target_filepath, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                for ds, n_dict in existing_data.items():
                    for n_val, methods in n_dict.items():
                        n_str = str(n_val)
                        if n_str not in TARGET_N_VALUES:
                            continue
                        for method, time_val in methods.items():
                            merged_data[ds][n_str][method] = time_val
            except json.JSONDecodeError:
                print(f"Warning: existing output is not valid JSON; starting fresh: {target_filepath}")

    # 3. Scan and merge supported input JSON files.
    search_pattern = os.path.join(input_dir, "*.json")
    for filepath in glob.glob(search_pattern):
        filename = os.path.basename(filepath)
        
        # Skip the output itself and FP16 result files.
        if filename == target_filename or "fp16" in filename:
            continue
            
        standard_method = get_standard_method_name(filename)
        if standard_method == "Unknown":
            continue

        print(f"Processing: {filename} -> method {standard_method}")

        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: skipping invalid JSON file: {filename}")
                continue

            for raw_ds_name, n_dict in data.items():
                ds_name = clean_dataset_name(raw_ds_name)
                
                for n_val, method_data in n_dict.items():
                    n_str = str(n_val) 
                    
                    if n_str not in TARGET_N_VALUES:
                        continue
                        
                    fastest_time = extract_fastest_time(method_data)
                    
                    if fastest_time is not None:
                        merged_data[ds_name][n_str][standard_method] = fastest_time

    # 4. Convert nested defaultdict objects into regular dictionaries.
    final_output = {}
    for k, v in merged_data.items():
        if v:
            final_output[k] = {nk: dict(nv) for nk, nv in v.items()}

    # 5. Write the merged result.
    with open(target_filepath, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    
    print(f"\nMerge complete. Output file: {target_filepath}")

if __name__ == "__main__":
    main()
