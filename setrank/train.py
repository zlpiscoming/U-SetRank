import model
import predata
from torch import optim, nn

def train(datas, args):
    SetRank = model.SetRank().to(args.device)
    optimizer = optim.Adam(SetRank.parameters(), lr=args.lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0

    for epoch in range(args.epochs):
        totalloss = 0
        for x, y in datas:
            x, y = x.to(args.device), y.to(args.device)
            model.train()
            logits = SetRank(x)
            loss = criteon(logits, y)
            totalloss = max(totalloss, loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoches = {}, loss = {}'.format(epoch, totalloss))

