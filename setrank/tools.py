import torch.nn as nn
import numpy as np
import numpy as np
import torch

class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, common_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

def P(ranks, E):
    # mapping rank to vector dimension N * E
    pass

def rFF(B):
    kinds = B.shape[0]
    B = B.flatten()
    model = MLP(B.shape[0], kinds)
    return model(B)


def Attn(Q, K, V):
    E  = Q.shape[1]
    K = K.T 
    tmp = nn.Softmax(Q*K/np.sqrt(E)).dim
    Res = tmp*V
    '''
        Q = Nq * E
        K.T = E * Nk
        V = Nk * E
        Res = Nq * E
    '''
    return Res

def Multihead(Q, K, V, h=1):
    Res = torch.ones([Q.shape[0],0])
    for i in range(h):
        _Q = Q[:, i:i + Q.shape[1]/h]
        _K = K[:, i:i + K.shape[1]/h]
        _V = V[:, i:i + V.shape[1]/h]
        Res = torch.cat([Res, Attn(_Q, _K, _V)])
    return Res

def layerNormalization(x):
    ln = torch.nn.LayerNorm(x.shape)
    return ln(x)

def generateRanks(datas):
    x, indices = torch.sort(datas, descending=True)
    return indices

def MAB(Q, K, V):
    B = layerNormalization(Q + Multihead(Q,K,V))
    return layerNormalization(B + rFF(B))

def MSAB(Q):
    return MAB(Q, Q, Q)

def IMSAB(Q, m):
    I = torch.rand([m, Q.shape[1]])
    '''
    this is the fake attention query vector 
    i cannot sure is made by random 
    '''
    H = MAB(I,Q,Q)
    return MAB(Q,H,H)

