
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """

    def __init__(self, in_dim, hidden_dim, num_layers, 
                 gcn_dropout, skip_connection=True,
                 out_dim=None, anneal=False):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.gcn_drop = nn.Dropout(gcn_dropout)
        self.skip_connection = skip_connection
        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.num_layers):
            input_dim = self.in_dim if layer == 0 else self.hidden_dim     
            #NEW: for FiLM
            if out_dim and layer == num_layers - 1:
                self.W.append(nn.Linear(input_dim, out_dim))
            else:
                if anneal:
                    self.hidden_dim = int(self.hidden_dim * (2 ** (-layer)))
                self.W.append(nn.Linear(input_dim, self.hidden_dim))

    def forward(self, node_reps, adj):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        # denom = adj.sum(1).unsqueeze(1) + 1

        # pool_mask = (adj.sum(2) + adj.sum(1)).eq(0)
        #QUES: The skip_connection only work if same in/hid/out dim
        #ANSW: Change to layer-wise skip_connect
        # raw_reps = node_reps
        for l in range(self.num_layers):
            raw_reps = node_reps
            # print(adj.size(), node_reps.size())
            Ax = adj.bmm(node_reps)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](node_reps)  # self loop
            AxW = AxW / (denom + 1e-8)

            gAxW = F.relu(AxW)
            node_reps = self.gcn_drop(gAxW) if l < self.num_layers - 1 else gAxW
            if self.skip_connection:
                node_reps += raw_reps
        # if self.skip_connection:
        #    node_reps += raw_reps
        return node_reps
