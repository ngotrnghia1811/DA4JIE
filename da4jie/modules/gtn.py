import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt

class AGIE_GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_layers, norm, num_bert_layers=24, device='cpu'):
        """
        num_edge: in_channel of GTLayer (#type of edges)
        num_channels: out_channel of GTLayer (C)
        w_in:
        w_out:
        num_class: 
        num_layers: number of GTN layers
        norm:
        """
        super(AGIE_GTN, self).__init__()
        self.device = device
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_layers = num_layers
        self.is_norm = norm
        self.num_bert_layers = num_bert_layers
        
        #! GTLayers for each layer
        layers = []
        for i in range(num_bert_layers):
            layers.append(GTLayer(num_edge, num_channels, first=True))
            break
        self.aug_layers = nn.ModuleList(layers*24)

        #! GTLayers for combine
        layers = []
        c_in = num_channels * num_bert_layers
        c_out = num_channels
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(c_in, c_out, first=True))
            else:
                layers.append(GTLayer(c_in, c_out, first=False))
        self.combine_layers = nn.ModuleList(layers)
        
        # GCN layers
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.bias = nn.Parameter(torch.Tensor(w_out))
        
        # FF layers
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self, X, H):
        # NxD * win x wout
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        return torch.mm(H.t(),X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        
        eye = torch.eye(H.shape[0]).to(self.device)
        # print(self.device,H.device, eye.device)
        if add == False:
            H = H*((eye==0).float())
        else:
            H = H*((eye==0).float()) + eye.float()
        deg = torch.sum(H, dim=1) + 1e-8
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*eye.float()
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H

    def forward(self, A, X):
        """
        #! Add Identity to A here instead of main
        #! currently, no minibatch training
        #!
        A (LxBxNxNxE) : input adj matrix
        X (BxNxD)   : input node feature matrix
        """
        # A = A.unsqueeze(0).permute(0,3,1,2) 
        bsz, n = A[0].size()[:2]


        A_aug = []
        for l in range(self.num_bert_layers):
            A_l = A[l]
            aug_graphs = []
            for b in range(bsz):
                A_b = A_l[b].unsqueeze(0).permute(0,3,1,2)
                # print(f'A_b: {A_b.size()}')
                H, _ = self.aug_layers[l](A_b)
                aug_graphs.append(H)
            aug_graphs = torch.cat(aug_graphs, dim=0).view(bsz, -1, n, n)
            # print(f'aug_graphs_{l}: {aug_graphs.size()}')
            A_aug.append(aug_graphs)

        A_aug = torch.cat(A_aug, dim=1).view(bsz, -1, n, n)
        # print(f'A_aug: {A_aug.size()}')

        X_ret = []
        H_ret = []
        for b in range(bsz):
            X_b = X[b]
            A_b = A_aug[b]
            # print(A_b.size(), X_b.size())
            A_b = A_b.unsqueeze(0)
            for i in range(self.num_layers):
                if i == 0:
                    H, _ = self.combine_layers[i](A_b)
                else:
                    H = self.normalization(H)
                    H, _ = self.combine_layers[i](A_b, H)

            for i in range(self.num_channels):
                if i==0:
                    X_ = F.relu(self.gcn_conv(X_b, H[i]))
                else:
                    X_tmp = F.relu(self.gcn_conv(X_b, H[i]))
                    X_ = torch.cat((X_,X_tmp), dim=1)
            X_ret.append(X_.unsqueeze(0))
            H_ret.append(H.unsqueeze(0))

        X_ret = torch.cat(X_ret, dim=0)
        H_ret = torch.cat(H_ret, dim=0)

        X_ret = self.linear1(X_ret)
        H_ret = torch.mean(H_ret, dim=1)
        
        # print(f'X_ret: {X_ret.size()}, H_ret: {H_ret.size()}')
        return H_ret, X_ret
        
        # #NEW: Aug Forward
        # aug_graphs = []
        # print(A[0].size())   # bsz x 1 x c x n x n
        # for l in range(self.num_bert_layers):
        #     A_l = A[l].unsqueeze(1).permute(0,1,4,2,3) 
        #     H, _ = self.aug_layers[l](A_l)
        #     # print(H.size())
        #     aug_graphs.append(H)
        # A_aug = torch.cat(aug_graphs, dim=0).view(bsz, self.num_bert_layers * self.num_channels, n, n)
        # print(f'A_aug: {A_aug.size()}')

        # #NEW: Combine Forward
        # X_ret = []
        # for b in range(bsz):
        #     A_b = A_aug[b]
        #     X_b = X[b]
        #     print(A_b.size(), X_b.size())
        #     for i in range(self.num_layers):
        #         if i == 0:
        #             H, _ = self.combine_layers[i](A_b)
        #         else:
        #             H = self.normalization(H)
        #             H, _ = self.combine_layers[i](A_b, H)
            
        #     #NOTE: GCN on meta-path H to compute node feature
        #     for i in range(self.num_channels):
        #         if i==0:
        #             X_ = F.relu(self.gcn_conv(X_b, H[i]))
        #         else:
        #             X_tmp = F.relu(self.gcn_conv(X_b, H[i]))
        #             X_ = torch.cat((X_,X_tmp), dim=1)
        #     X_ret.append(X_.unsqueeze(0))
        # X_ = torch.cat(X_ret, dim=0)
        # print(X_.size())
        # # Last FFW layers
        # X_ = self.linear1(X_)
        # return H, X_


class GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class,num_layers,norm, device='cpu'):
        """

        num_edge: in_channel of GTLayer (#type of edges)
        num_channels: out_channel of GTLayer (C)
        w_in:
        w_out:
        num_class: 
        num_layers: number of GTN layers
        norm:
        """
        super(GTN, self).__init__()
        self.device = device
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        
        # GTLayers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        
        # GCN layers
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.bias = nn.Parameter(torch.Tensor(w_out))
        
        # FF layers
        self.loss = nn.CrossEntropyLoss()
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self,X,H):
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        return torch.mm(H.t(),X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H*((torch.eye(H.shape[0], device=self.device)==0).float())
        else:
            H = H*((torch.eye(H.shape[0], device=self.device)==0).float()) + torch.eye(H.shape[0], device=self.device).float()
        deg = torch.sum(H, dim=1) + 1e-8
        deg_inv = deg.pow(-1) 
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*torch.eye(H.shape[0], device=self.device).float()
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H

    def forward(self, A, X, target_x, target):
        """
        #! Add Identity to A here instead of main
        #! currently, no minibatch training
        #!
        A (NxNxE) : input adj matrix
        X (NxD)   : input node feature matrix
        target_x (N): all node in train set
        target   (N): labels of above   
        """
        A = A.unsqueeze(0).permute(0,3,1,2) 
        
        #NOTE: Run each GTN layer (and collect phi, for ?)
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        
        #NOTE: GCN on meta-path H to compute node feature
        for i in range(self.num_channels):
            if i==0:
                X_ = F.relu(self.gcn_conv(X,H[i]))
            else:
                X_tmp = F.relu(self.gcn_conv(X,H[i]))
                X_ = torch.cat((X_,X_tmp), dim=1)

        # Last FFW layers
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        y = self.linear2(X_[target_x])
        loss = self.loss(y, target)
        return loss, y, Ws

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        """
        each GTLayer learn to softly select adj_matrices by 1x1conv
        a = F = conv(A) -> A = D A F = bmm(H_, a)
        """
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        """
        A : adj_matrices set 
        H_: prev GTLayer output ~ (k-1) meta-path
        """
        #! why return W (phi) ? no reason really 
        if self.first == True:
            a = self.conv1(A)
            b = self.conv2(A)
            # bs, c, n = a.size()[:3]
            # H = torch.bmm(a.view(bs*c,n,n),b.view(bs*c,n,n)).view(bs,c,n,n)
            H = torch.bmm(a,b)
            # W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_,a)
            # W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        W = None
        return H, W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #NOTE: 1x1 conv
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
        self.bias = None
        # self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        # bsz = A.size(0)
        # if self.batch:
            # A = torch.sum(A*F.softmax(self.weight.repeat(bsz,1,1,1,1), dim=2), dim=2)
        # else:
        # print(F.softmax(self.weight, dim=1))
        # print(A.size(), self.weight.size())
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A