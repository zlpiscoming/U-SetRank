import model
import predata
from torch import optim, nn
import tools
import torch

def train(datas, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SetRank = model.SetRank().to(device)
    optimizer = optim.Adam(SetRank.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        totalloss = 0
        for x, y in datas:
            x = x.to(device)
            y = y.to(device)
            ranks = SetRank(x)
            loss = tools.gernerateLoss(ranks, y)
            totalloss = max(totalloss, loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoches = {}, loss = {}'.format(epoch, totalloss))

