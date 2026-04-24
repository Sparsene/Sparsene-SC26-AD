import torch
import numpy as np
from scipy.sparse import *
import time
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('eva100/abalation/')
from gcn.mtest import *
from gcn.mdataset import *
from gat.mtest_gat import *
from gat.mdataset_gat import *

def norm(spmm):
    if spmm<10 :
        return '{:.2f}'.format(spmm)
    elif spmm < 100 :
        return '{:.1f}'.format(spmm)
    else:
        return '{:.0f}'.format(spmm)
    
# ablation-study
if __name__ == "__main__":
    gpu_device = torch.cuda.current_device()
    
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)
   

    hidden = [64, 128, 256, 512]

    epoches = 10
    
    dataset = [ 'AmazonProducts', 'reddit', 'amazon', 'IGB_medium']

    
    dataset = ['amazon']

    
    # #TF32
    file_name = './eva100/abalation/' + 'result-table6' + '.txt'
    with open(file_name, 'w') as file:
        file.write(gpu + ' : \n')
    

    for data in dataset:
        for dimN in hidden:
            res = ''
            
            if dimN == 64:
                res = res + '\multirow{4}{*}{' + data + '}'
            
            res = res + ' & ' + str(dimN) 
            
            # FP16
            # Initial-16x1
            inputInfo_16 = MGCN_dataset_m16()
            inputInfo_16.m_block_16_8_r(data, dimN)
            #res
            mgcn16_16 = test_fp16_v2_16(data, epoches, dimN, inputInfo_16)
            del inputInfo_16
            
             # with-16x1
            inputInfo_16_mr = MGCN_dataset_m16()
            inputInfo_16_mr.m_block_16_8_mr(data, dimN)
            # 16x1 v2
            mgcn16_16_mr = test_fp16_v2_16(data, epoches, dimN, inputInfo_16_mr)
            del inputInfo_16_mr
            
            # with-8x1
            inputInfo_8 = MGCN_dataset_m16()
            inputInfo_8.m_block_8_8_r(data, dimN)
            # 8x1 v2
            mgcn16_8_v2 = test_fp16_v2(data, epoches, dimN, inputInfo_8)
            del inputInfo_8
            
            
            # + MR
            inputInfo_8_mr = MGCN_dataset_m16()
            inputInfo_8_mr.m_block_8_8_mr(data, dimN)
            # 8x1+MR v2
            mgcn16_8_mr_v2 = test_fp16_v2(data, epoches, dimN, inputInfo_8_mr)
            del inputInfo_8_mr
            
            # v2
            res = res + ' & ' + str(norm(mgcn16_16)) + ' & ' 
            res = res + ' & ' + str(norm(mgcn16_16_mr))
            res = res + ' & ' + str(norm(mgcn16_8_v2)) 
            res = res + ' & ' + str(norm(mgcn16_8_mr_v2)) + ' & '

            
            # TF32
            # Initial-16x1
            inputInfo_16 = MGCN_dataset_m32()
            inputInfo_16.m_block_16_4_r(data, dimN)
            #res
            mgcn32_16 = test_tf32_v2_16(data, epoches, dimN, inputInfo_16)
            del inputInfo_16
            
            # with-16x1 +MR
            inputInfo_16_mr = MGCN_dataset_m32()
            inputInfo_16_mr.m_block_16_4_mr(data, dimN)
            # 16x1 v2
            mgcn32_16_mr = test_tf32_v2_16(data, epoches, dimN, inputInfo_16_mr)
            # del inputInfo_8
            
            # with-8x1
            inputInfo_8 = MGCN_dataset_m32()
            inputInfo_8.m_block_8_4_r(data, dimN)
            # 8x1 v2
            mgcn32_8_v2 = test_tf32_v2(data, epoches, dimN, inputInfo_8)
            # del inputInfo_8
            
            # with-8x1 +MR
            inputInfo_8_mr = MGCN_dataset_m32()
            inputInfo_8_mr.m_block_8_4_mr(data, dimN)
            # 8x1+MR v2
            mgcn32_8_mr_v2 = test_tf32_v2(data, epoches, dimN, inputInfo_8_mr)
            # del inputInfo_8_mr

            res = res + ' & ' + str(norm(mgcn32_16)) + ' & ' 
            res = res + ' & ' + str(norm(mgcn32_16_mr))
            res = res + ' & ' + str(norm(mgcn32_8_v2))
            res = res + ' & ' + str(norm(mgcn32_8_mr_v2)) + ' & '
         
            
        


            
            # GAT - FP16
            # Initial-16x1
            inputInfo_16_gat = MGCN_dataset_m16_gat()
            inputInfo_16_gat.m_block_16_8_r(data, dimN)
            #res
            mgcn16_16_gat = test_fp16_v2_16_gat(data, epoches, dimN, inputInfo_16_gat)
            del inputInfo_16_gat
            
            # Initial-16x1 + MR
            inputInfo_16_mr_gat = MGCN_dataset_m16_gat()
            inputInfo_16_mr_gat.m_block_16_8_mr(data, dimN)
            #res
            mgcn16_16_mr_gat = test_fp16_v2_16_gat(data, epoches, dimN, inputInfo_16_mr_gat)
            del inputInfo_16_mr_gat
            
            # with-8x1
            inputInfo_8_gat = MGCN_dataset_m16_gat()
            inputInfo_8_gat.m_block_8_16_r(data, dimN)
            # 8x1 v2
            mgcn16_8_v2_gat = test_fp16_v2_gat(data, epoches, dimN, inputInfo_8_gat)
            del inputInfo_8_gat
            
            # + MR
            inputInfo_8_mr_gat = MGCN_dataset_m16_gat()
            inputInfo_8_mr_gat.m_block_8_16_mr(data, dimN)
            # 8x1+MR v2
            mgcn16_8_mr_v2_gat = test_fp16_v2_gat(data, epoches, dimN, inputInfo_8_mr_gat)
            del inputInfo_8_mr_gat
            

            res = res + ' & ' + str(norm(mgcn16_16_gat)) + ' & ' 
            res = res + ' & ' + str(norm(mgcn16_16_mr_gat))
            res = res + ' & ' + str(norm(mgcn16_8_v2_gat))
            res = res + ' & ' + str(norm(mgcn16_8_mr_v2_gat))  + ' & '      
       
            # GAT - TF32
            # Initial-16x1
            inputInfo_16_gat = MGCN_dataset_m32_gat()
            inputInfo_16_gat.m_block_16_8_r(data, dimN)
            #res
            mgcn32_16_gat = test_tf32_v2_16_gat(data, epoches, dimN, inputInfo_16_gat)
            del inputInfo_16_gat
               
            # 16x1 + MR            
            inputInfo_16_mr_gat = MGCN_dataset_m32_gat()
            inputInfo_16_mr_gat.m_block_16_8_mr(data, dimN)
            #res
            mgcn32_16_mr_gat = test_tf32_v2_16_gat(data, epoches, dimN, inputInfo_16_mr_gat)
            del inputInfo_16_mr_gat
            
            # with-8x1
            inputInfo_8_gat = MGCN_dataset_m32_gat()
            inputInfo_8_gat.m_block_8_16_r(data, dimN)
            # 8x1 v2
            mgcn32_8_v2_gat = test_tf32_v2_gat(data, epoches, dimN, inputInfo_8_gat)
            del inputInfo_8_gat
            
            # + MR
            inputInfo_8_mr_gat = MGCN_dataset_m32_gat()
            inputInfo_8_mr_gat.m_block_8_16_mr(data, dimN)
            # 8x1+MR v2
            mgcn32_8_mr_v2_gat = test_tf32_v2_gat(data, epoches, dimN, inputInfo_8_mr_gat)
            del inputInfo_8_mr_gat
            

            res = res + ' & ' + str(norm(mgcn32_16_gat)) + ' & ' 
            res = res + ' & ' + str(norm(mgcn32_16_mr_gat))
            res = res + ' & ' + str(norm(mgcn32_8_v2_gat))
            res = res + ' & ' + str(norm(mgcn32_8_mr_v2_gat)) 
            
            res = res + ' \\\\ ' 
            if dimN == 512:
                res = res + ' \\\\ ' 
            with open(file_name, 'a') as file:
                file.write(res + '\n')
        print(data + ' is successed!')
        print()
    print('all success!')

    
  