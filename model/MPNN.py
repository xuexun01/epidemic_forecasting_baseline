import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv

class MPNN_LSTM(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout):
        super(MPNN_LSTM, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        self.nhid = nhid
        self.nfeat = nfeat
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.rnn1 = nn.LSTM(2*nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)
        
        self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc2 = nn.Linear( nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, adj, x):
        lst = list()
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
        skip = x.view(-1,self.window,self.n_nodes,self.nfeat)#self.batch_size
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)#self.batch_size*self.n_nodes
        
        x = self.relu(self.conv1(x, adj,edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = torch.cat(lst, dim=1)
        
        #--------------------------------------
        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes
        
        x, (hn1, cn1) = self.rnn1(x)
        
        
        out2, (hn2,  cn2) = self.rnn2(x)
        
        x = torch.cat([hn1[0,:,:],hn2[0,:,:]], dim=1)
        skip = skip.reshape(skip.size(0),-1)
                
        x = torch.cat([x,skip], dim=1)
        #--------------------------------------
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)
        
        
        return x