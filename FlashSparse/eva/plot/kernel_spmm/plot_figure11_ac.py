import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

base_128 = pd.read_csv(project_dir + '/result/Baseline/spmm/all_baseline_128.csv')
base_256 = pd.read_csv(project_dir + '/result/Baseline/spmm/all_baseline_256.csv')
temp = pd.merge(base_128, base_256, on='dataSet', how='inner')  
data_filter = temp[['dataSet']]

num_0 =0
num_1 =0
edge = 100000
for index, row in base_128.iterrows():
    if row['num_nodes'] > edge:
        num_0+=1
    else:
        num_1+=1

speedup = dict()
speedup['hidden'] = []
speedup['baseline'] = []
speedup['speedup'] = []
speedup['dataset'] = []

# First summarize cusaprse
cusparse = dict()
df = pd.read_csv(project_dir + '/dataset/data_filter.csv')
for index, row in df.iterrows():
    cusparse[row.iloc[0]] = dict()
for index, row in base_128.iterrows():
    cusparse[row.iloc[0]][128] = row['cusparse']
for index, row in base_256.iterrows():
    cusparse[row.iloc[0]][256] = row['cusparse']

# Compute FlashSparse's speedup normolized with cuSPARSE
hidden = [128, 256]
for dimN in hidden:
    base_var = f"base_{dimN}"  
    base_df = globals()[base_var]
    base_df= base_df[['dataSet']]
    # FlashSparse_fp16
    fs_fp16 = pd.read_csv(project_dir + '/result/FlashSparse/spmm/spmm_fp16_' + str(dimN) + '.csv')
    fs_fp16 = pd.merge(base_df, fs_fp16, on='dataSet', how='inner')  
    for index, row in fs_fp16.iterrows():
        speedup['baseline'].append('FlashSparse_fp16')
        speedup['dataset'].append(row[0])
        if row['num_nodes'] <= edge:
            speedup['hidden'].append(str(dimN)+'-small')
            speedup['speedup'].append(round( (cusparse[row[0]][dimN]/min(row['8_1_map'], row['8_1_balance'])) , 2) )
        else:
            speedup['hidden'].append(str(dimN)+'-large')
            speedup['speedup'].append(round( (cusparse[row[0]][dimN]/min(row['8_1_map'], row['8_1_balance'])) , 2) )
        
    # FlashSparse_tf32
    fs_tf32 = pd.read_csv(project_dir + '/result/FlashSparse/spmm/spmm_tf32_' + str(dimN) + '.csv')
    fs_tf32 = pd.merge(base_df, fs_tf32, on='dataSet', how='inner')  
    for index, row in fs_tf32.iterrows():
        speedup['baseline'].append('FlashSparse_tf32')
        speedup['dataset'].append(row[0])
        if row['num_nodes'] <= edge:
            speedup['hidden'].append(str(dimN)+'-small')
            speedup['speedup'].append(round( (cusparse[row[0]][dimN]/min(row['8_1_map'], row['8_1_balance'])) , 2) )
        else:
            speedup['hidden'].append(str(dimN)+'-large')
            speedup['speedup'].append(round( (cusparse[row[0]][dimN]/min(row['8_1_map'], row['8_1_balance'])) , 2) )

# Compute Baselines' speedup normolized with cuSPARSE
for index, row in base_128.iterrows():
    # rode
    speedup['baseline'].append('RoDe')
    speedup['dataset'].append(row[0])
    if row['num_nodes'] <= edge:
        speedup['hidden'].append('128-small')
        speedup['speedup'].append(round( (row['cusparse']/row['rode']) , 2) )
    else:
        speedup['hidden'].append('128-large')
        speedup['speedup'].append(round( (row['cusparse']/row['rode']) , 2) )
        
    # gespmm
    speedup['baseline'].append('GE-SpMM')
    speedup['dataset'].append(row[0])
    if row['num_nodes'] <= edge:
        speedup['hidden'].append('128-small')
        speedup['speedup'].append(round( (row['cusparse']/row['gespmm']) , 2) )
    else:
        speedup['hidden'].append('128-large')
        speedup['speedup'].append(round( (row['cusparse']/row['gespmm']) , 2) )
              
    # sputnik
    speedup['baseline'].append('Sputnik')
    speedup['dataset'].append(row[0])
    if row['num_nodes'] <= edge:
        speedup['hidden'].append('128-small')
        speedup['speedup'].append(round( (row['cusparse']/row['sputnik']) , 2) )
    else:
        speedup['hidden'].append('128-large')
        speedup['speedup'].append(round( (row['cusparse']/row['sputnik']) , 2) )

    # advisor
    speedup['baseline'].append('GNNAdvisor')
    speedup['dataset'].append(row[0])
    if row['num_nodes'] <= edge:
        speedup['hidden'].append('128-small')
        speedup['speedup'].append(round( (row['cusparse']/row['advisor']) , 2) )
    else:
        speedup['hidden'].append('128-large')
        speedup['speedup'].append(round( (row['cusparse']/row['advisor']) , 2) )
    # dtc
    speedup['baseline'].append('DTC-SpMM')
    speedup['dataset'].append(row[0])
    if row['num_nodes'] <= edge:
        speedup['hidden'].append('128-small')
        speedup['speedup'].append(round( (row['cusparse']/row['dtc']) , 2) )
    else:
        speedup['hidden'].append('128-large')
        speedup['speedup'].append(round( (row['cusparse']/row['dtc']) , 2) )
        
    # tcgnn
    speedup['baseline'].append('TC-GNN')
    speedup['dataset'].append(row[0])
    if row['num_nodes'] <= edge:
        speedup['hidden'].append('128-small')
        speedup['speedup'].append(round( (row['cusparse']/row['tcgnn']) , 2) )
    else:
        speedup['hidden'].append('128-large')
        speedup['speedup'].append(round( (row['cusparse']/row['tcgnn']) , 2) )   



# Baseline
for index, row in base_256.iterrows():

    # rode
    speedup['baseline'].append('RoDe')
    speedup['dataset'].append(row[0])
    if row['num_nodes'] <= edge:
        speedup['hidden'].append('256-small')
        speedup['speedup'].append(round( (row['cusparse']/row['rode']) , 2) )
    else:
        speedup['hidden'].append('256-large')
        speedup['speedup'].append(round( (row['cusparse']/row['rode']) , 2) )

    # # gespmm
    speedup['baseline'].append('GE-SpMM')
    speedup['dataset'].append(row[0])
    if row['num_nodes'] <= edge:
        speedup['hidden'].append('256-small')
        speedup['speedup'].append(round( (row['cusparse']/row['gespmm']) , 2) )
    else:
        speedup['hidden'].append('256-large')
        speedup['speedup'].append(round( (row['cusparse']/row['gespmm']) , 2) )
        
    # sputnik
    speedup['baseline'].append('Sputnik')
    speedup['dataset'].append(row[0])
    if row['num_nodes'] <= edge:
        speedup['hidden'].append('256-small')
        speedup['speedup'].append(round( (row['cusparse']/row['sputnik']) , 2) )
    else:
        speedup['hidden'].append('256-large')
        speedup['speedup'].append(round( (row['cusparse']/row['sputnik']) , 2) )


    # advisor
    speedup['baseline'].append('GNNAdvisor')
    speedup['dataset'].append(row[0])
    if row['num_nodes'] <= edge:
        speedup['hidden'].append('256-small')
        speedup['speedup'].append(round( (row['cusparse']/row['advisor']) , 2) )
    else:
        speedup['hidden'].append('256-large')
        speedup['speedup'].append(round( (row['cusparse']/row['advisor']) , 2) )


    # dtc
    speedup['baseline'].append('DTC-SpMM')
    speedup['dataset'].append(row[0])
    if row['num_nodes'] <= edge:
        speedup['hidden'].append('256-small')
        speedup['speedup'].append(round( (row['cusparse']/row['dtc']) , 2) )
    else:
        speedup['hidden'].append('256-large')
        speedup['speedup'].append(round( (row['cusparse']/row['dtc']) , 2) ) 
        
    # tcgnn
    speedup['baseline'].append('TC-GNN')
    speedup['dataset'].append(row[0])
    if row['num_nodes'] <= edge:
        speedup['hidden'].append('256-small')
        speedup['speedup'].append(round( (row['cusparse']/row['tcgnn']) , 2) )
    else:
        speedup['hidden'].append('256-large')
        speedup['speedup'].append(round( (row['cusparse']/row['tcgnn']) , 2) )   


res = pd.DataFrame(speedup)
res.loc[res['speedup'] > 5, 'speedup'] = 5
palette = sns.color_palette('Greens')
mycolor = { 'FlashSparse_fp16':'royalblue', 'FlashSparse_tf32':'cornflowerblue', 
           'Sputnik': 'silver', 'RoDe': 'red', 
            'GNNAdvisor':'khaki', 'TC-GNN':'limegreen', 'DTC-SpMM': 'plum',
            'GE-SpMM':'orange'}

plt.figure(figsize=(12, 4)) 
desired_order = ['128-small', '128-large', '256-small', '256-large']
g = sns.boxplot(x='hidden', y='speedup', hue='baseline', data=res, 
                palette=mycolor, linewidth=1, legend=False, gap=0.15, width=0.9, order=desired_order)

from matplotlib.patches import Patch
legend_handles = [Patch(color=color, label=label) for label, color in mycolor.items()]
plt.legend(handles=legend_handles, title="", loc='upper right', fontsize=10, title_fontsize=12)

# plt.axhline(y=1, color='black', linestyle='--', linewidth = 1.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# sns.despine(left=False, right=True, top=True)
g.set_ylabel('')
g.set_xlabel('')


plt.savefig(project_dir + '/eva/plot/kernel_spmm/figure11.png', dpi=800)
plt.clf()
