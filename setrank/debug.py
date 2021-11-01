import model
from torchsummary import summary
import torch

def checkmodel(Model):
    for name, parameters in Model.named_parameters():
        print(name, ':', parameters.size())

if __name__ == '__main__':
    # device = 'cuda:0'
    # Model = model.SetRank()
    # print(Model)
    # summary(Model, (200, 200))
    # datas = torch.rand([5])
    # print(datas)
    # datas, ranks = torch.sort(datas,descending=True)
    # print(datas)
    # print(ranks)
    datas = torch.rand([5,4])
    print(datas)
    datas = datas.flatten()
    print(datas)