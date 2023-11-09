import torch.nn as nn


class SIRDcell(nn.Module):
    def __init__(self):
        super(SIRDcell, self).__init__()
    
    def forword(self, dS, dI, dR, dD):
        pass

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()