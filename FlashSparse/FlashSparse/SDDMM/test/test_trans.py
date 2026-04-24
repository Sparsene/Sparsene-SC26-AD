import torch
import MagicsphereGAT
import MagicsphereBlock


col = torch.tensor([1,11,14,0,1,2,7,1,3,5,8,3,10,1,3,10,5,9,0,11,12,0],dtype=torch.int32)
row = torch.tensor([0,3,7,8,11,11,13,13,14, 15,16,18,20,21,21,22,22],dtype=torch.int32)
row1,col1,value1 = MagicsphereBlock.blockProcess8_4_1(row,col)
value2 = value1.numpy()
for i in range(8):
    for j in range(4):
        print(value2[ i*4 + j], end=" ")
    print()
    if i%8==7 :
        print()

shape = value1.shape
templete = torch.ones(shape, dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
row1=row1.to(device)
col1=col1.to(device)
value1=value1.to(device)
templete = templete.to(device)

output=MagicsphereGAT.trans_gat_tf32(16, row1, col1, value1, templete, 12)
outputcpu=output.to('cpu').numpy()

for i in range(8):
    for j in range(4):
        print(outputcpu[ i*4 + j], end=" ")
    print()
    if i%8==7 :
        print()
