import pandas as pd
import os
import argparse

#              
METHOD_FILES = {
    "SparseNE": [
        "sc-ad-gcn_e2e_dtc_base_26_128-512.csv",
        "sc-ad-gcn_e2e_dtc_multibind_26_128-512.csv",
        "sc-ad-gcn_e2e_dtc_strict_lb_26_128-512.csv",
        "sc-ad-gcn_e2e_srbcrs_base_26_128-512.csv",
        "sc-ad-gcn_e2e_srbcrs_16x8_26_128-512.csv",
        "sc-ad-gcn_e2e_srbcrs_16x8_multibind_26_128-512.csv",
        "sc-ad-gcn_e2e_srbcrs_16x8_strict_lb_26_128-512.csv"
    ],
    "FlashSparse": ["sc-ad-gcn_e2e_flashsparse_26_128-512.csv"],
    "DTC-SpMM": ["sc-ad-gcn_e2e_dtc_origin_26_128-512.csv"],
    "DGL": ["sc-ad-gcn_e2e_dgl_26_128-512.csv"],
    "PyG": ["sc-ad-gcn_e2e_pyg_26_128-512.csv"]
}

#    N    ，     N        'hidden'  
N_COLUMN = "hidden" 
TARGET_N_VALUES = [128, 256, 512]

def load_and_process_method(method_name, file_list, data_dir):
    """
               CSV         
    """
    df_list = []
    for file in file_list:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"  :       {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            #           N 
            if N_COLUMN in df.columns:
                df = df[df[N_COLUMN].isin(TARGET_N_VALUES)]
                df_list.append(df[['dataset', N_COLUMN, 'time_per_epoch_ms']])
            else:
                print(f"  :    {file}        {N_COLUMN}")
        except Exception as e:
            print(f"     {file_path}    : {e}")
            
    if not df_list:
        return pd.DataFrame(columns=['dataset', N_COLUMN, 'time_per_epoch_ms'])
        
    return pd.concat(df_list, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description="Calculate average speedup of SparseNE against other baselines.")
    parser.add_argument(
        "-d", "--data_dir", 
        type=str, 
        default="sparsene/end2end/result", 
        help="    CSV        (  : sparsene/end2end/result)"
    )
    args = parser.parse_args()
    
    data_dir = args.data_dir
    print(f"         : {data_dir}\n")

    # 1.    SparseNE   
    sparsene_df = load_and_process_method("SparseNE", METHOD_FILES["SparseNE"], data_dir)
    if sparsene_df.empty:
        print("  :          SparseNE   ！         。")
        return
        
    sparsene_best = sparsene_df.groupby(['dataset', N_COLUMN], as_index=False)['time_per_epoch_ms'].min()
    sparsene_best.rename(columns={'time_per_epoch_ms': 'time_SparseNE'}, inplace=True)
    
    final_df = sparsene_best.copy()
    
    # 2.      Baselines
    baselines = ["PyG", "DGL", "DTC-SpMM", "FlashSparse"]
    for baseline in baselines:
        baseline_df = load_and_process_method(baseline, METHOD_FILES[baseline], data_dir)
        
        if not baseline_df.empty:
            baseline_best = baseline_df.groupby(['dataset', N_COLUMN], as_index=False)['time_per_epoch_ms'].min()
            baseline_best.rename(columns={'time_per_epoch_ms': f'time_{baseline}'}, inplace=True)
            final_df = pd.merge(final_df, baseline_best, on=['dataset', N_COLUMN], how='left')
        else:
            final_df[f'time_{baseline}'] = None

    # 3.               
    speedup_cols = []
    for baseline in baselines:
        time_col = f'time_{baseline}'
        speedup_col = f'Avg_Speedup_vs_{baseline}'
        if time_col in final_df.columns:
            # Baseline    / SparseNE    =    
            final_df[speedup_col] = final_df[time_col] / final_df['time_SparseNE']
            speedup_cols.append(speedup_col)
    
    # 4.   N (hidden)     ，    Baseline       
    if not speedup_cols:
        print("         Baseline   。")
        return
        
    avg_speedup_df = final_df.groupby(N_COLUMN)[speedup_cols].mean().reset_index()
    
    #              
    avg_speedup_df[speedup_cols] = avg_speedup_df[speedup_cols].round(2)
    
    # 5.       
    print("="*80)
    print("===    N (Hidden)           ===")
    print("="*80)
    
    #         
    print(avg_speedup_df.to_string(index=False))

if __name__ == "__main__":
    main()