import numpy as np

# =====      =====
N_total = 684          #      
sample_sizes = [10, 20, 50, 200]  #        
top_ks = [10, 20, 50, 200]        #   k      
num_trials = 100000    #     

# =====    =====
np.random.seed(42)  #      

results = {k: {n: 0 for n in sample_sizes} for k in top_ks}

for trial in range(num_trials):
    #          
    sample = np.random.choice(N_total, max(sample_sizes), replace=False)
    
    for sample_size in sample_sizes:
        sample_subset = sample[:sample_size]
        for k in top_ks:
            #    sample_subset         k    
            count_in_topk = np.sum(sample_subset < k)
            results[k][sample_size] += count_in_topk / k  #   

#       
print("        k    （     ）：")
for k in top_ks:
    row = [f"{results[k][n]/num_trials:.3f}" for n in sample_sizes]
    print(f"top-{k}: {' '.join(row)}")
