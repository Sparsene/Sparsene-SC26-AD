import sys
# sys.path.append('./eva100/end2end/agnn_no_pre')
from mypyg.agnn_pyg import AGNN, train
from mypyg.mdataset import *

    
def test(data, epoches, layers, featuredim, hidden, classes):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #start_time = time.time()
        inputInfo = MGCN_dataset(data, featuredim, classes)

        inputInfo.to(device)
        model = AGNN(inputInfo.num_features, hidden, inputInfo.num_classes,  layers).to(device)
        train(inputInfo ,model, 10)
        torch.cuda.synchronize()
        start_time = time.time()
        train(inputInfo ,model, epoches)
        #         
        torch.cuda.synchronize()
        end_time = time.time()
        execution_time = end_time - start_time
        #         （   ）
        # print(round(execution_time,4))
        return round(execution_time,4)
    
# if __name__ == "__main__":
#     dataset = 'blog'
#     test(dataset, 100, 3, 64, 512, 10)