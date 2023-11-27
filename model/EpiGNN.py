import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ELU()
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)

class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, tanhalpha=1):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        # embedding [batchsize, hidden_dim]
        nodevec1 = self.linear1(embedding)
        nodevec2 = self.linear2(embedding)
        nodevec1 = self.alpha * nodevec1
        nodevec2 = self.alpha * nodevec2
        nodevec1 = torch.tanh(nodevec1)
        nodevec2 = torch.tanh(nodevec2)
        
        adj = torch.bmm(nodevec1, nodevec2.permute(0, 2, 1))-torch.bmm(nodevec2, nodevec1.permute(0, 2, 1))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj

class ConvBranch(nn.Module):
    def __init__(self, m, in_channels, out_channels, kernel_size, dilation_factor, hidP=1, isPool=True):
        super().__init__()
        self.m = m
        self.isPool = isPool
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size,1), dilation=(dilation_factor,1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        if self.isPool:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        #self.activate = nn.Tanh()
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.isPool:
            x = self.pooling(x)
        x = x.view(batch_size, -1, self.m)
        return x

class RegionAwareConv(nn.Module):
    def __init__(self, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()
        self.P = P
        self.m = m
        self.k = k
        self.hidP = hidP
        self.conv_l1 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=3, dilation_factor=1, hidP=self.hidP)
        self.conv_l2 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=5, dilation_factor=1, hidP=self.hidP)
        self.conv_p1 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=3, dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_p2 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=5, dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_g = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=self.P, dilation_factor=1, hidP=None, isPool=False)
        self.activate = nn.Tanh()
    
    def forward(self, x):
        x = x.view(-1, 1, self.P, self.m)
        # local pattern
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)
        # periodic pattern
        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)
        # global
        x_global = self.conv_g(x)
        # concat and activate
        x = torch.cat([x_local, x_period, x_global], dim=1).permute(0, 2, 1)
        x = self.activate(x)
        return x


def getLaplaceMat(batch_size, m, adj):
    i_mat = torch.eye(m).to(adj.device)
    i_mat = i_mat.unsqueeze(0)
    o_mat = torch.ones(m).to(adj.device)
    o_mat = o_mat.unsqueeze(0)
    i_mat = i_mat.expand(batch_size, m, m)
    o_mat = o_mat.expand(batch_size, m, m)
    adj = torch.where(adj>0, o_mat, adj)
    d_mat = torch.sum(adj, dim=2) # attention: dim=2
    d_mat = d_mat.unsqueeze(2)
    d_mat = d_mat + 1e-12
    d_mat = torch.pow(d_mat, -0.5)
    d_mat = d_mat.expand(d_mat.shape[0], d_mat.shape[1], d_mat.shape[1])
    d_mat = i_mat * d_mat

    # laplace_mat = d_mat * adj * d_mat
    laplace_mat = torch.bmm(d_mat, adj)
    laplace_mat = torch.bmm(laplace_mat, d_mat)
    return laplace_mat


class EpiGNN(nn.Module):
    def __init__(self, n_nodes, hist_window, pred_window, n_layer, dropout, hidden_R, hidden_A, hidden_P, n_kernel, n_layer_GCN, kernel_size, residual, extra, highway, origin_adj, adj):
        super().__init__()
        self.n_nodes = n_nodes
        self.hist_window = hist_window
        self.n_layer = n_layer
        self.hidR = hidden_R
        self.hidA = hidden_A
        self.hidP = hidden_P
        self.n_kernel = n_kernel
        self.kernel_size = kernel_size
        self.n_layer_GCN = n_layer_GCN
        self.residual = residual
        self.hw = highway
        self.dropout = nn.Dropout(dropout)
        self.adj = adj

        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)
        if extra:
            self.extra = True
        else:
            self.extra = False

        # Feature embedding
        self.hidR = self.n_kernel*4*self.hidP + self.n_kernel
        self.backbone = RegionAwareConv(P=self.hist_window, m=self.n_nodes, k=self.n_kernel, hidP=self.hidP)

        # global
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.t_enc = nn.Linear(1, self.hidR)

        # local
        self.degree = torch.sum(origin_adj, dim=-1)
        self.s_enc = nn.Linear(1, self.hidR)

        # external resources
        self.external_parameter = nn.Parameter(torch.FloatTensor(self.n_nodes, self.n_nodes), requires_grad=True)

        # Graph Generator and GCN
        self.d_gate = nn.Parameter(torch.FloatTensor(self.n_nodes, self.n_nodes), requires_grad=True)
        self.graphGen = GraphLearner(self.hidR)
        self.GNNBlocks = nn.ModuleList([GraphConvLayer(in_features=self.hidR, out_features=self.hidR) for i in range(self.n_layer_GCN)])

        # prediction
        if self.residual == 0:
            self.output = nn.Linear(self.hidR*2, pred_window)
        else:
            self.output = nn.Linear(self.hidR*(self.n_layer_GCN+1), pred_window)
        self.init_weights()
     

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
    

    def forward(self, inputs, isEval=False):
        batch_size = inputs.shape[0] # batchsize, w, m

        # step 1: Use multi-scale convolution to extract feature embedding (SEFNet => RAConv).
        temp_emb = self.backbone(inputs)

        # step 2: generate global transmission risk encoding.
        query = self.WQ(temp_emb) # batch, N, hidden
        query = self.dropout(query)
        key = self.WK(temp_emb)
        key = self.dropout(key)
        attn = torch.bmm(query, key.transpose(1, 2))
        #attn = self.leakyrelu(attn)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1)
        attn = attn.unsqueeze(2)
        t_enc = self.t_enc(attn)
        t_enc = self.dropout(t_enc)

        # step 3: generate local transmission risk encoding.
        # print(self.degree.shape) [self.m]
        d = self.degree.unsqueeze(1)
        s_enc = self.s_enc(d)
        s_enc = self.dropout(s_enc)

        # Three embedding fusion.
        feat_emb = temp_emb + t_enc + s_enc

        # step 4: Region-Aware Graph Learner
        # apply Graph Learner to generate a graph
        d_mat = torch.mm(d, d.permute(1, 0))
        d_mat = torch.mul(self.d_gate, d_mat)
        d_mat = torch.sigmoid(d_mat)
        spatial_adj = torch.mul(d_mat, self.adj)
        adj = self.graphGen(temp_emb)
        
        # if additional information => fusion
        adj = adj + spatial_adj

        # get laplace adjacent matrix
        laplace_adj = getLaplaceMat(batch_size, self.n_nodes, adj)
        
        # Graph Convolution Network
        node_state = feat_emb
        node_state_list = []
        for layer in self.GNNBlocks:
            node_state = layer(node_state, laplace_adj)
            node_state = self.dropout(node_state)
            node_state_list.append(node_state)

        # Final prediction
        node_state = torch.cat([node_state, feat_emb], dim=-1)
        res = self.output(node_state).squeeze(2)
        # highway means autoregressive model
        if self.hw > 0:
            z = inputs[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.n_nodes)
            res = res + z
        
        # if evaluation, return some intermediate results
        if isEval:
            imd = (adj, attn)
        else:
            imd = None
        return res, imd
