import torch
import numpy as np
from scipy.sparse import *
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('eva100/abalation/gcn/')
from eva100.abalation.gat.mtest_gat import *
from eva100.abalation.gat.mdataset_gat import *
import csv

# ablation-study
if __name__ == "__main__":
    #          GPU   
    gpu_device = torch.cuda.current_device()
    
    #    GPU      
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)
   
    dataset = ['Reddit2', 'ovcar', 'amazon','amazon0505',
            'yelp', 'sw620', 'dd',
            'HR_NO', 'HU_NO', 'ell', 'GitHub',
            'artist', 'comamazon', 
            'yeast', 'blog', 'DGraphFin', 'reddit', 'ogb', 'AmazonProducts', 'IGB_medium', 'IGB_large']
    

    hidden = [64, 128, 256, 512, 1024]
    epoches = 1000
    
    dataset = ['blog', 'reddit', 'amazon', 'IGB_medium','AmazonProducts']
    # dataset = ['blog', 'dd']
    # hidden = [64, 128]
    head= ['Dataset', 'Dim', 'Initial-16x1', 'with-8x1-v2', 'with-8x1-MR-v2', 'Speedup1-v2', 'Speedup2-v2']
    
    # #TF32
    # # CSV     
    csv_file = './eva100/abalation/gcn/' + 'tf32' + '-H100.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
    
    #TF32
    for data in dataset:
        for dimN in hidden:
            res = []
            res.append(data)
            res.append(str(dimN))
            
            # Initial-16x1
            inputInfo_16 = MGCN_dataset_m32_gat()
            inputInfo_16.m_block_16_4_r(data, dimN)
            #res
            mgcn32_16 = test_tf32_v2_16_gat(data, epoches, dimN, inputInfo_16)
            del inputInfo_16
            res.append(str(mgcn32_16))
            
            # with-8x1
            inputInfo_8 = MGCN_dataset_m32_gat()
            inputInfo_8.m_block_8_4_r(data, dimN)
            # 8x1 v2
            mgcn32_8_v2 = test_tf32_v2_gat(data, epoches, dimN, inputInfo_8)
            # del inputInfo_8
            
            # + MR
            inputInfo_8_mr = MGCN_dataset_m32_gat()
            inputInfo_8_mr.m_block_8_4_mr(data, dimN)
            # 8x1+MR v2
            mgcn32_8_mr_v2 = test_tf32_v2_gat(data, epoches, dimN, inputInfo_8_mr)
            # del inputInfo_8_mr

            res.append(str(mgcn32_8_v2))
            res.append(str(mgcn32_8_mr_v2))
            
            res.append(str(round((mgcn32_16/mgcn32_8_v2),4)))
            res.append(str(round((mgcn32_16/mgcn32_8_mr_v2),4)))
            print("Speed up v2-8x1 :" + str(round((mgcn32_16/mgcn32_8_v2),4)))
            print("Speed up v2-MR :" + str(round((mgcn32_16/mgcn32_8_mr_v2),4)))
            
            #    CSV   
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(res)
        print(data + ' is successed!')
        print()
    # print(f'{csv_file}     ！')
    # print()
    
    # # FP16
    # CSV     
    csv_file = './eva100/abalation/gcn/' + 'fp16' + '-H100.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
        
    # FP16
    for data in dataset:
        for dimN in hidden:
            res = []
            res.append(data)
            res.append(str(dimN))
            
            # Initial-16x1
            inputInfo_16 = MGCN_dataset_m16_gat()
            inputInfo_16.m_block_16_8_r(data, dimN)
            #res
            mgcn16_16 = test_fp16_v2_16_gat(data, epoches, dimN, inputInfo_16)
            del inputInfo_16
            res.append(str(mgcn16_16))
            
            # with-8x1
            inputInfo_8 = MGCN_dataset_m16_gat()
            inputInfo_8.m_block_8_8_r(data, dimN)
            # 8x1 v2
            mgcn16_8_v2 = test_fp16_v2_gat(data, epoches, dimN, inputInfo_8)
            del inputInfo_8
            
            # + MR
            inputInfo_8_mr = MGCN_dataset_m16_gat()
            inputInfo_8_mr.m_block_8_8_mr(data, dimN)
            # 8x1+MR v2
            mgcn16_8_mr_v2 = test_fp16_v2_gat(data, epoches, dimN, inputInfo_8_mr)
            del inputInfo_8_mr
            
            # v2
            res.append(str(mgcn16_8_v2))
            res.append(str(mgcn16_8_mr_v2))
            # v2 speedup
            res.append(str(round((mgcn16_16/mgcn16_8_v2),4)))
            res.append(str(round((mgcn16_16/mgcn16_8_mr_v2),4)))
            print("Speed up v2-8x1 :" + str(round((mgcn16_16/mgcn16_8_v2),4)))
            print("Speed up v2-MR :" + str(round((mgcn16_16/mgcn16_8_mr_v2),4)))
            
            #    CSV   
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(res)
        print(data + ' is successed!')
        print()
    print(f'{csv_file}     ！')
    print()

    
  