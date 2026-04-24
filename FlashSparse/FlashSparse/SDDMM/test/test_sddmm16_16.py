import torch
import MagicsphereBlock
import MagicsphereGAT
import MagicsphereGAT2
col = torch.tensor([1,3,11,1,7,1,2,5,5,7,0,0,3,8,9,11,1,6,2,7,3,6],dtype=torch.int32)
row = torch.tensor([0,3,5,6,8,8,8,10,11,14,16,18,20,20,22,22,22],dtype=torch.int32)
d=(row[1:] - row[:-1]).tolist()
degree = torch.tensor(d,dtype=torch.half)
row,col,value = MagicsphereBlock.blockProcess16_8(row,col,degree)

print(row)
print(col)
print(value)


dimM = 32
dimMori=30
dimN = 80
lhs = torch.ones((dimMori, dimN),dtype=torch.float16)
rhs = torch.ones((dimMori, dimN),dtype=torch.float16)
shape = (30, dimN)
rhs1 = torch.full(shape, 4,dtype=torch.float16)
# print(rhs1)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# row=row.to(device)
# col=col.to(device)
# value=value.to(device)
# lhs=lhs.to(device)
# rhs=rhs.to(device)
# rhs1=rhs1.to(device)
# w0=w0.to(device)
# w1=w1.to(device)

output, _=MagicsphereGAT2.forward_gen_16(dimN, dimM, row, col, value, lhs, rhs1,32,100)
# outputcpu=output.to('cpu').numpy()

for i in range(4):
    print(output[i*4:i*4+4])
