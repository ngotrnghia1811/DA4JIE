from re import S
import torch
import torch.nn as nn

import numpy as np

def token_lens_to_offsets(token_lens):
    """Map token lengths to first word piece indices, used by the sentence
    encoder.
    :param token_lens (list): token lengths (word piece numbers)
    :return (list): first word piece indices (offsets)
    """
    max_token_num = max([len(x) for x in token_lens])
    offsets = []
    for seq_token_lens in token_lens:
        seq_offsets = [0]
        for l in seq_token_lens[:-1]:
            seq_offsets.append(seq_offsets[-1] + l)
        offsets.append(seq_offsets + [-1] * (max_token_num - len(seq_offsets)))
    return offsets

def token_lens_to_idxs(token_lens):
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs = []
    for seq_token_lens in token_lens:
        seq_idxs = []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.append([offset, offset + token_len])
            offset += token_len
        seq_idxs.extend([[-1, 0]] * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
    return idxs, max_token_num, max_token_len


def graphs_to_node_idxs(graphs):
    """
    :param graphs (list): A list of Graph objects.
    :return: mention/trigger index matrix, mask tensor, max number, and max length
    """
    entity_idxs = []
    trigger_idxs = []

    max_entity_num = max(max(graph.entity_num for graph in graphs), 1)
    max_trigger_num = max(max(graph.trigger_num for graph in graphs), 1)
    max_entity_len = max(max([e[1] - e[0] for e in graph.entities] + [1])
                         for graph in graphs)
    max_trigger_len = max(max([t[1] - t[0] for t in graph.triggers] + [1])
                          for graph in graphs)
    for bid, graph in enumerate(graphs):
        seq_entity_idxs = []
        seq_trigger_idxs = []

        for entity in graph.entities:
            seq_entity_idxs.append([entity[0], entity[1]])
        seq_entity_idxs.extend([[0, 1]] * (max_entity_num - graph.entity_num))
        entity_idxs.append(seq_entity_idxs)

        for trigger in graph.triggers:
            seq_trigger_idxs.append([trigger[0], trigger[1]])

        seq_trigger_idxs.extend([[0, 1]] * (max_trigger_num - graph.trigger_num))

        trigger_idxs.append(seq_trigger_idxs)

    return (
        entity_idxs, max_entity_num, max_entity_len,
        trigger_idxs, max_trigger_num, max_trigger_len
    )

def graphs_to_node_idxs_with_mask(graphs):
    """
    :param graphs (list): A list of Graph objects.
    :return: entity/trigger index matrix, mask tensor, max number, and max length
    """
    entity_idxs, entity_masks = [], []
    trigger_idxs, trigger_masks = [], []
    max_entity_num = max(max(graph.entity_num for graph in graphs), 1)
    max_trigger_num = max(max(graph.trigger_num for graph in graphs), 1)
    max_entity_len = max(max([e[1] - e[0] for e in graph.entities] + [1])
                         for graph in graphs)
    max_trigger_len = max(max([t[1] - t[0] for t in graph.triggers] + [1])
                          for graph in graphs)
    for graph in graphs:
        seq_entity_idxs, seq_entity_masks = [], []
        seq_trigger_idxs, seq_trigger_masks = [], []
        for entity in graph.entities:
            entity_len = entity[1] - entity[0]
            seq_entity_idxs.extend([i for i in range(entity[0], entity[1])])
            seq_entity_idxs.extend([0] * (max_entity_len - entity_len))
            seq_entity_masks.extend([1.0 / entity_len] * entity_len)
            seq_entity_masks.extend([0.0] * (max_entity_len - entity_len))
        seq_entity_idxs.extend([0] * max_entity_len * (max_entity_num - graph.entity_num))
        seq_entity_masks.extend([0.0] * max_entity_len * (max_entity_num - graph.entity_num))
        entity_idxs.append(seq_entity_idxs)
        entity_masks.append(seq_entity_masks)

        for trigger in graph.triggers:
            trigger_len = trigger[1] - trigger[0]
            seq_trigger_idxs.extend([i for i in range(trigger[0], trigger[1])])
            seq_trigger_idxs.extend([0] * (max_trigger_len - trigger_len))
            seq_trigger_masks.extend([1.0 / trigger_len] * trigger_len)
            seq_trigger_masks.extend([0.0] * (max_trigger_len - trigger_len))
        seq_trigger_idxs.extend([0] * max_trigger_len * (max_trigger_num - graph.trigger_num))
        seq_trigger_masks.extend([0.0] * max_trigger_len * (max_trigger_num - graph.trigger_num))
        trigger_idxs.append(seq_trigger_idxs)
        trigger_masks.append(seq_trigger_masks)

    return (
        entity_idxs, entity_masks, max_entity_num, max_entity_len,
        trigger_idxs, trigger_masks, max_trigger_num, max_trigger_len,
    )

def graphs_to_label_idxs(graphs, max_entity_num=-1, max_trigger_num=-1,
                         relation_directional=False,
                         symmetric_relation_idxs=None):
    """Convert a list of graphs to label index and mask matrices
    :param graphs (list): A list of Graph objects.
    :param max_entity_num (int) Max entity number (default = -1).
    :param max_trigger_num (int) Max trigger number (default = -1).
    """
    if max_entity_num == -1:
        max_entity_num = max(max([g.entity_num for g in graphs]), 1)
    if max_trigger_num == -1:
        max_trigger_num = max(max([g.trigger_num for g in graphs]), 1)
    (
        batch_entity_idxs, batch_entity_mask,
        batch_trigger_idxs, batch_trigger_mask,
        batch_relation_idxs, batch_relation_mask,
        batch_role_idxs, batch_role_mask
    ) = [[] for _ in range(8)]
    for graph in graphs:
        (
            entity_idxs, entity_mask, trigger_idxs, trigger_mask,
            relation_idxs, relation_mask, role_idxs, role_mask,
        ) = graph.to_label_idxs(max_entity_num, max_trigger_num,
                                relation_directional=relation_directional,
                                symmetric_relation_idxs=symmetric_relation_idxs)
        batch_entity_idxs.append(entity_idxs)
        batch_entity_mask.append(entity_mask)
        batch_trigger_idxs.append(trigger_idxs)
        batch_trigger_mask.append(trigger_mask)
        batch_relation_idxs.append(relation_idxs)
        batch_relation_mask.append(relation_mask)
        batch_role_idxs.append(role_idxs)
        batch_role_mask.append(role_mask)
    return (
        batch_entity_idxs, batch_entity_mask,
        batch_trigger_idxs, batch_trigger_mask,
        batch_relation_idxs, batch_relation_mask,
        batch_role_idxs, batch_role_mask
    )


def tag_paths_to_spans(paths, token_nums, vocab):
    """Convert predicted tag paths to a list of spans (entity mentions or event
    triggers).
    :param paths: predicted tag paths.
    :return (list): a list (batch) of lists (sequence) of spans.
    """
    batch_mentions = []
    itos = {i: s for s, i in vocab.items()}
    for i, path in enumerate(paths):
        mentions = []
        cur_mention = None
        path = path.tolist()[:token_nums[i].item()]
        for j, tag in enumerate(path):
            tag = itos[tag]
            if tag == 'O':
                prefix = tag = 'O'
            else:
                prefix, tag = tag.split('-', 1)

            if prefix == 'B':
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = [j, j + 1, tag]
            elif prefix == 'I':
                if cur_mention is None:
                    # treat it as B-*
                    cur_mention = [j, j + 1, tag]
                elif cur_mention[-1] == tag:
                    cur_mention[1] = j + 1
                else:
                    # treat it as B-*
                    mentions.append(cur_mention)
                    cur_mention = [j, j + 1, tag]
            else:
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = None
        if cur_mention:
            mentions.append(cur_mention)
        batch_mentions.append(mentions)
    return batch_mentions

#NOTE: AGIE
def convert_token_len_to_assignment(token_lens, max_piece_len, max_token_len):
    # token_len : [1,2,1,3]
    bsz = len(token_lens)
    asgm_mat = torch.zeros(bsz, max_piece_len, max_token_len)
    for b in range(bsz):
        prev = 0
        for i, l in enumerate(token_lens[b]):
            # shift 1 for CLS
            for j in range(prev, prev+l):
            # asgm_mat[b][i+1][l] = 1
                asgm_mat[b][j+1][i] = 1
            prev += l
    return asgm_mat

def convert_span_to_assignment(spans, max_token_len, max_span_len):
    # span : [0,2] , [4,5] , [0,1],[0,1],[0,1]...
    bsz = len(spans)
    asgm_mat = torch.zeros(bsz, max_token_len, max_span_len)
    # print(spans)
    # print(spans.size())
    for b in range(bsz):
        for i, l in enumerate(spans[b]):
            if i == 0 and l == [0,1]:
                continue

            for j in range(l[0], l[1]): # each pos of span
                try:
                    asgm_mat[b][j][i] = 1
                except:
                    print(spans[b])
                    print(max_token_len, max_span_len)
                    asgm_mat[b][j][i] = 1
    return asgm_mat

def convert_dep_head_to_adj(dep_heads):
    # dep_heads : [1,4,0,2,3]
    bsz, max_token_len = dep_heads.size()
    adj_mat = torch.zeros(bsz, max_token_len, max_token_len)
    for b in range(bsz):
        for i, j in enumerate(dep_heads[b]):
            j -= 1
            if j < 0:
                continue
            try:
                adj_mat[b][i][j] = 1
            except:
                print(adj_mat.size(), b, i, j)
                adj_mat[b][i][j] = 1  # raise the same error
    return adj_mat

class Linears(nn.Module):
    """Multiple linear layers with Dropout."""

    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        return inputs

class GRDA(nn.Module):

    def __init__(self, config):
        super().__init__()

        args = config.grda
        self.device = args.device
        self.batch_size = args.batch_size
        self.lambda_gan = args.grda_lambda
        self.num_domain = args.num_domain
        self.num_sample_d = args.num_sample_d # sample how many vertices for training D
        self.num_sample_g = args.num_sample_g # sample how many vertices for training G

        self.A = args.dom_adj_matrix

        self.Discr_ffn = Linears(args.Ddim,
                                 dropout_prob=config.linear_dropout,
                                 bias=config.linear_bias,
                                 activation=config.linear_activation)

        self.Gdom_ffn = Linears(args.Gdim,
                                 dropout_prob=config.linear_dropout,
                                 bias=config.linear_bias,
                                 activation=config.linear_activation)
        
        self.Discr_criterion = nn.BCEWithLogitsLoss()
        self.Gdom_criterion = nn.BCEWithLogitsLoss()

    def forward(self, dom_cond_reprs):
        
        # cond_reprs = {k:i.unsqueeze(0) for k, v in dom_cond_reprs.items() if 'cond' in k for i in v}
        # for k,v in cond_reprs.items():
            # print(k, v.size())
        # cond_reprs = torch.cat(cond_reprs.values(), dim=0)

        # Gdom_reprs = [i.unsqueeze(0) for k, v in dom_cond_reprs.items() if 'dom' in k for i in v]
        # Gdom_reprs = torch.cat(Gdom_reprs, dim=0)
 
        cond_reprs = dom_cond_reprs['labeled_cond'] + dom_cond_reprs['unlabeled_cond']

        Gdom_reprs = dom_cond_reprs['labeled_dom'] + dom_cond_reprs['unlabeled_dom']

        c = [self.Discr_ffn(r) for r in cond_reprs]
        # print(f'lent: {c[0].size(1)}, ltrig: {c[1].size(1)}, uent: {c[2].size(1)}, utrig:{c[3].size(1)}')
        loss_Discr = self.loss_Discr(c)

        d = [self.Gdom_ffn(r) for r in Gdom_reprs]
        loss_Gdom = self.loss_Gdom(d)

        loss_enc_gan = - self.lambda_gan * loss_Discr
        return loss_Discr, loss_Gdom, loss_enc_gan
        
    def loss_Gdom(self, d):
        # d ~ #num * bsz * #ent/trig * hdim
        #NEW: Loss to train Graph_Dom Encoder
        sub_graph = self.sub_graph(self.num_sample_g)
        errorG = torch.zeros((1,)).to(self.device)
        sample_size = self.num_sample_g

        for i in range(sample_size):
            v_i = sub_graph[i]
            for j in range(i + 1, sample_size):
                v_j = sub_graph[j]
                # label = torch.tensor(self.A[v_i][v_j]).to(self.device).float()
                label = torch.full((self.batch_size,), self.A[v_i][v_j],
                                   device=self.device,).float()
                #QUES: d ~ T*B, C, why * bsz? maybe change to same as loss_D ?
                # output = (d[v_i * self.batch_size] * d[v_j * self.batch_size]).sum()
                # output = (d[v_i] * d[v_j]).sum(1)
                vi_norm = d[v_i] / d[v_i].norm(dim=2, keepdim=True)
                vj_norm = d[v_j] / d[v_j].norm(dim=2, keepdim=True)
                output = torch.bmm(vi_norm,
                                    vj_norm.permute(0,2,1)
                                    ).mean(2).mean(1)
                errorG += self.Gdom_criterion(output, label)

        errorG /= sample_size * (sample_size - 1) / 2
        return errorG

    def loss_Discr(self, c):
        # c ~ #dom * bsz * #ent/trig * hdim
        #NEW: Loss to train Discriminator
        sub_graph = self.sub_graph(self.num_sample_d)

        errorD_connected = torch.zeros((1,)).to(self.device)  # .double()
        errorD_disconnected = torch.zeros((1,)).to(self.device)  # .double()

        count_connected = 1e-8
        count_disconnected = 1e-8

        for i in range(self.num_sample_d):
            v_i = sub_graph[i]
            for j in range(i + 1, self.num_sample_d):
                v_j = sub_graph[j]
                label = torch.full((self.batch_size,), self.A[v_i][v_j],
                                   device=self.device,).float()
                # dot product
                if v_i == v_j:
                    #QUES: This dont happen ? d ~ T,B,C
                    idx = torch.randperm(self.batch_size)
                    output = (c[v_i][idx] * c[v_j]).sum(1)
                else:
                    # output = (c[v_i] * c[v_j]).sum(1)
                    vi_norm = c[v_i] / c[v_i].norm(dim=2, keepdim=True)
                    vj_norm = c[v_j] / c[v_j].norm(dim=2, keepdim=True)
                    output = torch.bmm(vi_norm,
                                       vj_norm.permute(0,2,1)
                                      ).mean(2).mean(1)
                if self.A[v_i][v_j]:  # connected
                    errorD_connected += self.Discr_criterion(output, label)
                    count_connected += 1
                else:
                    errorD_disconnected += self.Discr_criterion(output, label)
                    count_disconnected += 1

        errorD = 0.5 * (
            errorD_connected / count_connected
            + errorD_disconnected / count_disconnected
        )
        #QUES: this is a loss balance ?
        return errorD * self.num_domain

    def sub_graph(self, sample_size):
        #NEW: Mixture policy for sampling node

        #NEW: policy 1: random sample
        if np.random.randint(0, 2) == 0:
            return np.random.choice(
                self.num_domain, size=sample_size, replace=False
            )

        #NEW: policy 2: pick from random choosen connected sub-graph
        # subsample a chain (or multiple chains in graph)
        left_nodes = self.num_sample_d  # num node left need to sample
        choosen_node = []
        vis = np.zeros(self.num_domain)  # visited
        while left_nodes > 0:
            # Repeat rand_walk sample till left_nodes = 0
            chain_node, node_num = self.rand_walk(vis, left_nodes)
            choosen_node.extend(chain_node)
            left_nodes -= node_num

        return choosen_node
    
    def rand_walk(self, vis, left_nodes):
        #NEW: Uniform sampled random walk

        # Init chain
        chain_node = []
        node_num = 0     #

        # Init random start node
        node_index = np.where(vis==0)[0]  # ~ range(num_dom)
        st = np.random.choice(node_index)
        vis[st] = 1
        chain_node.append(st)
        left_nodes -= 1
        node_num += 1

        cur_node = st
        while left_nodes > 0:
            nx_node = -1

            node_to_choose = np.where(vis==0)[0]
            num = node_to_choose.shape[0]
            node_to_choose = np.random.choice(
                node_to_choose, num, replace=False
            )

            for i in node_to_choose:
                if cur_node != i:
                    # have an edge and not yet visit
                    if self.A[cur_node][i] and not vis[i]:
                        nx_node = i
                        vis[nx_node] = 1
                        chain_node.append(nx_node)
                        left_nodes -= 1
                        node_num += 1
                        break
            if nx_node >= 0:
                cur_node = nx_node
            else:
                break

        return chain_node, node_num