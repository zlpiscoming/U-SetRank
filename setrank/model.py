import torch.nn as nn
import numpy as np
import tools

class SetRank(nn.Module):
    def __init__(self, maxlen=200, kinds = 'MSAB', Nb = 6):
        super(SetRank, self).__init__()
        self.premodel = 'bm25'
        self.kinds = kinds
        self.E = maxlen
        self.Nb = Nb
        '''
        this is the feature len
        '''

    def forward(self, x):
        for i in range(self.Nb):
            if self.kinds == 'MSAB':
                x = tools.MSAB(x)
            else:
                x = tools.IMSAB(x)
            x = tools.layerNormalization(x)
        x = tools.rFF(x)
        x = tools.layerNormalization(x)
        x = tools.rFF(x)
        outputs = tools.generateRanks(x)
        return outputs
        '''
        after this x will become N * 1 while the 1 is prob
        '''

