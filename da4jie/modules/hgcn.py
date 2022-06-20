# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GCN
from .module_utils import get_full_adj, get_pruned_adj
from ..util import *
from .graph import Graph

from ..models.model_utils import GRDA, Linears


class HierachicalGCN(nn.Module):
    """
    #NOTE: For FourIE
    Hierarchical GCN module operated on two levels of information:
    (i) word level (over dependency tree of the sentence).
    (ii) task level (over task tree of the sentence).
    """

    def __init__(self, args, num_types, config=None):
        super(HierachicalGCN, self).__init__()
        self.args = args
        self.bert_dim = args.gcn_dim
        self.num_types = num_types
        self.relation_trans = nn.Sequential(nn.Linear(4 * self.args.gcn_dim, self.args.gcn_dim), nn.ReLU())
        self.role_trans = nn.Sequential(nn.Linear(4 * self.args.gcn_dim, self.args.gcn_dim), nn.ReLU())
        # ********* task embedding *********
        self.event_embed = nn.Parameter(torch.zeros(self.args.task_embed_dim, 1)).cuda()
        self.entity_embed = nn.Parameter(torch.zeros(self.args.task_embed_dim, 1)).cuda()
        self.role_embed = nn.Parameter(torch.zeros(self.args.task_embed_dim, 1)).cuda()
        self.relation_embed = nn.Parameter(torch.zeros(self.args.task_embed_dim, 1)).cuda()
        # ********* task GCN **********
        self.task_gcn = GCN(
            in_dim=self.args.gcn_dim + self.args.task_embed_dim,
            hidden_dim=self.args.gcn_dim + self.args.task_embed_dim,
            num_layers=2,
            gcn_dropout=self.args.gcn_drop
        )
        # ******** task-specific & cross-task fushion **********
        self.fuse_W_hat = nn.Linear(self.args.gcn_dim + self.args.task_embed_dim,
                                    self.args.gcn_dim + self.args.task_embed_dim)
        self.fuse_W_tilde = nn.Linear(self.args.gcn_dim + self.args.task_embed_dim,
                                      self.args.gcn_dim + self.args.task_embed_dim)
        # ******** label GCN ***************
        self.label_gcn = GCN(
            in_dim=self.args.type_embed_dim,
            hidden_dim=self.args.type_embed_dim,
            num_layers=2,
            gcn_dropout=self.args.gcn_drop,
            skip_connection=False
        )
        self.init_A = torch.arange(0, self.num_types * self.num_types).float().view(self.num_types,
                                                                                    self.num_types).cuda()

        self.init_embeds()

        self.config = config
        #NEW: dom embedding
        self.dom_emb_type = config.dom_emb_type
        self.dom_embedding = nn.Embedding(len(config.dom_emb_type), 
                                        config.dom_emb_size[0])
        self.dom_ffn = Linears(config.dom_emb_size,
                            dropout_prob=config.linear_dropout,
                            bias=config.linear_bias,
                            activation=config.linear_activation)
        #NEW: Currently concat [repr, dom]
        self.combine_ffn = Linears([config.dom_emb_size[-1] + self.bert_dim, self.bert_dim],
                                dropout_prob=config.linear_dropout,
                                bias=config.linear_bias,
                                activation=config.linear_activation)


    def init_embeds(self):
        torch.nn.init.xavier_normal_(self.event_embed.data)
        torch.nn.init.xavier_normal_(self.entity_embed.data)
        torch.nn.init.xavier_normal_(self.role_embed.data)
        torch.nn.init.xavier_normal_(self.relation_embed.data)

    def forward(self, gcn_input, predict):
        batch = gcn_input['batch']
        input_reps = gcn_input['bert_outputs']
        graphs_info = gcn_input['graphs_info']
        # ********** node info *******************
        task_adj = self.get_task_adj(graphs_info)  # [batch size, num task nodes, num nodes]
        # ****************************************
        #NEW: ret for grda
        h_Lm_tilde, ret = self.get_task_specific_reps(gcn_input, predict)  # [batch size, num nodes, wgcn dim + task_embed]
        h_Lm_hat = self.task_gcn(h_Lm_tilde, task_adj)
        f_gate = self.get_fgate(h_Lm_tilde, h_Lm_hat)
        h_Lm = f_gate * h_Lm_tilde + (1 - f_gate) * h_Lm_hat  # [batch size, num task nodes, word dim]
        # ********** extract reps ****************
        T = graphs_info['max_trigger_num']
        E = graphs_info['max_entity_num']
        #NOTE: Representations of graphs
        trigger_reprs = h_Lm[:, :T, :]  # [batch size, max trigger num, word dim]
        entity_reprs = h_Lm[:, T: T + E, :]  # [batch size, max entity num, word dim]
        te_reprs = h_Lm[:, T + E: T + E + T * E, :]  # [batch size, max trigger num * max entity num, word dim]
        ee_reprs = h_Lm[:, T + E + T * E:, :]  # [batch size, max entity num * max entity num, word dim]

        # entity type scores
        entity_type_scores = gcn_input['entity_type_ffn'](entity_reprs)
        # mention type scores
        mention_type_scores = gcn_input['mention_type_ffn'](entity_reprs)
        # event type scores
        event_type_scores = gcn_input['event_type_ffn'](trigger_reprs)
        # relation type scores
        relation_type_scores = gcn_input['relation_type_ffn'](ee_reprs)
        use_entity_type = True
        if use_entity_type:
            trigger_num = gcn_input['graphs_info']['max_trigger_num']
            bert_outputs = gcn_input['bert_outputs']
            batch_size = bert_outputs.shape[0]
            if predict:
                entity_type_scores_softmax = entity_type_scores.softmax(dim=2)
                entity_type_scores_softmax = entity_type_scores_softmax.repeat(1, trigger_num, 1)
                te_reprs = torch.cat([te_reprs, entity_type_scores_softmax], dim=2)
            else:
                entity_types = batch.entity_type_idxs.view(batch_size, -1)
                entity_types = torch.clamp(entity_types, min=0)
                entity_types_onehot = bert_outputs.new_zeros(*entity_types.size(),
                                                             gcn_input['entity_type_num'])
                entity_types_onehot.scatter_(2, entity_types.unsqueeze(-1), 1)
                entity_types_onehot = entity_types_onehot.repeat(1, trigger_num, 1)
                te_reprs = torch.cat([te_reprs, entity_types_onehot], dim=2)
        role_type_scores = gcn_input['role_type_ffn'](te_reprs)
        if not predict:
            score_vectors = {
                'entity': entity_type_scores,  # [batch size, num entities, num entity types]
                'event': event_type_scores,  # [batch size, num events, num event types]
                'relation': relation_type_scores,  # [batch size, num relations, num relation types]
                'role': role_type_scores  # [batch size, num roles, num role types]
            }
            # for k,v in score_vectors.items():
            #     if type(v) == torch.tensor:
            #         print(k, v.size)
            #     else:
            #         print(k, v)
            pattern_loss = self.compute_pattern_loss(gcn_input, score_vectors)
        else:
            pattern_loss = 0

        return entity_type_scores, mention_type_scores, event_type_scores, relation_type_scores, role_type_scores, pattern_loss, ret

    def compute_pattern_loss(self, gcn_input, score_vectors):
        batch_size = score_vectors['entity'].shape[0]
        type_embeddings = torch.cat(
            [gcn_input['entity_type_ffn'].layers[-1].weight,
             gcn_input['event_type_ffn'].layers[-1].weight,
             gcn_input['relation_type_ffn'].layers[-1].weight,
             gcn_input['role_type_ffn'].layers[-1].weight
             ],
            dim=0
        )  # [num entity types + num event types + num relation types + num role types, embed dim]
        type_embeddings = type_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch size, num types, embed dim]
        predicted_adj = self.decode_label_adj(gcn_input, score_vectors)  # [batch size, num types, num types]
        gold_adj = self.get_label_adj(gcn_input)  # [batch size, num types, num types]
        # print(f'predicted_adj: {predicted_adj}')
        # print(f'gold_adj: {gold_adj}')
        predicted_graph = self.label_gcn(type_embeddings, predicted_adj)  # [batch size, num types, embed dim]
        gold_graph = self.label_gcn(type_embeddings, gold_adj)  # [batch size, num types, embed dim]
        pattern_loss = torch.sum((predicted_graph - gold_graph) ** 2)
        return pattern_loss

    def get_fgate(self, h_li_tilde, h_li_hat):
        return torch.sigmoid(self.fuse_W_tilde(h_li_tilde) \
                             + self.fuse_W_hat(h_li_hat))

    def differentiable_matrix_filling(self, row, column):
        dim_A = self.init_A.shape[0]
        # x = float(row * dim_A + `column)
        x = row * dim_A + column
        A_x = torch.exp( - ((self.init_A - x) ** 2) * self.args.temperature)
        return A_x

    def decode_label_adj(self, gcn_input, score_vectors):
        entity_type_scores = score_vectors['entity']  # [batch size, max entity num]
        event_type_scores = score_vectors['event']  # [batch size, max trigger num]
        relation_type_scores = score_vectors['relation']  # [batch size, max entity num * max entity num]
        role_type_scores = score_vectors['role']  # [batch size, max trigger num * max entity num]

        entity_type_num = gcn_input['entity_type_ffn'].layers[-1].weight.shape[0]
        event_type_num = gcn_input['event_type_ffn'].layers[-1].weight.shape[0]
        relation_type_num = gcn_input['relation_type_ffn'].layers[-1].weight.shape[0]
        role_type_num = gcn_input['role_type_ffn'].layers[-1].weight.shape[0]

        entity_type_start = 0
        event_type_start = entity_type_num
        relation_type_start = entity_type_num + event_type_num
        role_type_start = entity_type_num + event_type_num + relation_type_num

        batch_size = gcn_input['bert_outputs'].shape[0]
        graphs_info = gcn_input['graphs_info']
        max_trigger_num = graphs_info['max_trigger_num']
        max_entity_num = graphs_info['max_entity_num']

        all_label_adjs = []
        for bid in range(batch_size):
            actual_trigger_num = graphs_info['trigger_nums'][bid]
            actual_entity_num = graphs_info['entity_nums'][bid]
            label_adj = torch.ones_like(self.init_A).cuda() * 1e-12  # [num types, num types]
            # capturing occurrence of (entity type, relation type) and (entity type, entity type) pairs
            for entity_id1 in range(actual_entity_num):
                for entity_id2 in range(entity_id1):
                    relation_id = entity_id1 * max_entity_num + entity_id2
                    relation_type = torch.nn.functional.gumbel_softmax(relation_type_scores[bid][relation_id], tau=self.args.tau,
                                                                       hard=False).unsqueeze(0).matmul(
                        torch.arange(0, relation_type_num).float().unsqueeze(1).cuda())
                    # relation_type = torch.argmax(relation_type_scores[bid][relation_id])
                    # relation_type = relation_type_scores[bid][relation_id].argmax().item()
                    if relation_type > 0:
                        entity1_type = torch.nn.functional.gumbel_softmax(entity_type_scores[bid][entity_id1], tau=self.args.tau,
                                                                          hard=False).unsqueeze(0).matmul(
                            torch.arange(0, entity_type_num).float().unsqueeze(1).cuda())
                        # entity1_type = torch.argmax(entity_type_scores[bid][entity_id1])
                        # entity1_type = entity_type_scores[bid][entity_id1].argmax().item()
                        entity2_type = torch.nn.functional.gumbel_softmax(entity_type_scores[bid][entity_id2], tau=self.args.tau,
                                                                          hard=False).unsqueeze(0).matmul(
                            torch.arange(0, entity_type_num).float().unsqueeze(1).cuda())
                        # entity2_type = entity_type_scores[bid][entity_id2].argmax().item()
                        # assert entity1_type * entity2_type > 0
                        # connect entity type to the relation type that it participates in
                        # - for entity type 1:
                        # label_adj[entity_type_start + entity1_type][relation_type_start + relation_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=entity_type_start + entity1_type,
                            column=relation_type_start + relation_type
                        )
                        # label_adj[relation_type_start + relation_type][entity_type_start + entity1_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=relation_type_start + relation_type,
                            column=entity_type_start + entity1_type
                        )
                        # - for entity type 2:
                        # label_adj[entity_type_start + entity2_type][relation_type_start + relation_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=entity_type_start + entity2_type,
                            column=relation_type_start + relation_type
                        )
                        # label_adj[relation_type_start + relation_type][entity_type_start + entity2_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=relation_type_start + relation_type,
                            column=entity_type_start + entity2_type
                        )
            # capturing occurrence of (event, entity type) and (entity type, entity type) pairs
            for trigger_id in range(actual_trigger_num):
                for entity_id in range(actual_entity_num):
                    role_id = trigger_id * max_entity_num + entity_id
                    role_type = torch.nn.functional.gumbel_softmax(role_type_scores[bid][role_id], tau=self.args.tau,
                                                                   hard=False).unsqueeze(0).matmul(
                        torch.arange(0, role_type_num).float().unsqueeze(1).cuda())
                    # role_type = torch.argmax(role_type_scores[bid][role_id])
                    # role_type = role_type_scores[bid][role_id].argmax().item()
                    if role_type > 0:
                        event_type = torch.nn.functional.gumbel_softmax(event_type_scores[bid][trigger_id], tau=self.args.tau,
                                                                        hard=False).unsqueeze(0).matmul(
                            torch.arange(0, event_type_num).float().unsqueeze(1).cuda())
                        # event_type = torch.argmax(event_type_scores[bid][trigger_id])
                        # event_type = event_type_scores[bid][trigger_id].argmax().item()
                        entity_type = torch.nn.functional.gumbel_softmax(entity_type_scores[bid][entity_id], tau=self.args.tau,
                                                                         hard=False).unsqueeze(0).matmul(
                            torch.arange(0, entity_type_num).float().unsqueeze(1).cuda())
                        # entity_type = torch.argmax(entity_type_scores[bid][entity_id])
                        # entity_type = entity_type_scores[bid][entity_id].argmax().item()
                        # assert event_type * entity_type > 0
                        # connect entity type to the event type that it serves as an argument
                        # label_adj[entity_type_start + entity_type][event_type_start + event_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=entity_type_start + entity_type,
                            column=event_type_start + event_type
                        )
                        # label_adj[event_type_start + event_type][entity_type_start + entity_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=event_type_start + event_type,
                            column=entity_type_start + entity_type
                        )
                        # connect entity type to the role that it plays in this event
                        # label_adj[entity_type_start + entity_type][role_type_start + role_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=entity_type_start + entity_type,
                            column=role_type_start + role_type
                        )
                        # label_adj[role_type_start + role_type][entity_type_start + entity_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=role_type_start + role_type,
                            column=entity_type_start + entity_type
                        )
                        # connect event type to the role type to capture possible argument types of each event type
                        # label_adj[event_type_start + event_type][role_type_start + role_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=event_type_start + event_type,
                            column=role_type_start + role_type
                        )
                        # label_adj[role_type_start + role_type][event_type_start + event_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=role_type_start + role_type,
                            column=event_type_start + event_type
                        )
            all_label_adjs.append(label_adj)
        final_adj = torch.stack(all_label_adjs, dim=0)  # [batch size, num types, num types]
        return final_adj

    def get_label_adj(self, gcn_input):
        batch_size = gcn_input['bert_outputs'].shape[0]
        max_trigger_num = gcn_input['graphs_info']['max_trigger_num']
        max_entity_num = gcn_input['graphs_info']['max_entity_num']

        entity_type_num = gcn_input['entity_type_ffn'].layers[-1].weight.shape[0]
        event_type_num = gcn_input['event_type_ffn'].layers[-1].weight.shape[0]
        relation_type_num = gcn_input['relation_type_ffn'].layers[-1].weight.shape[0]
        role_type_num = gcn_input['role_type_ffn'].layers[-1].weight.shape[0]

        entity_type_start = 0
        event_type_start = entity_type_num
        relation_type_start = entity_type_num + event_type_num
        role_type_start = entity_type_num + event_type_num + relation_type_num

        entity_type_idxs = gcn_input['batch'].entity_type_idxs.view(batch_size, -1)  # [batch size, max entity num]
        event_type_idxs = gcn_input['batch'].event_type_idxs.view(batch_size, -1)  # [batch size, max trigger num]
        relation_type_idxs = gcn_input['batch'].relation_type_idxs.view(batch_size,
                                                                        -1)  # [batch size, max entity num * max entity num],
        # ->>> each segment of (max entity num) indicates the relations of the current entity with other entities

        role_type_idxs = gcn_input['batch'].role_type_idxs.view(batch_size,
                                                                -1)  # [batch size, max trigger num * max entity num],
        # ->>> each segment of (max entity num) indicates the argument roles of all entities in the sentence w.r.t the current trigger.
        all_label_adjs = []
        for bid in range(batch_size):
            label_adj = torch.ones_like(self.init_A).cuda() * 1e-12  # [num types, num types]
            # print('Before filling: {}'.format(torch.sum(label_adj)))
            # capturing occurrence of (entity type, relation type) and (entity type, entity type) pairs
            for entity_id1 in range(max_entity_num):
                for entity_id2 in range(entity_id1):
                    relation_id = entity_id1 * max_entity_num + entity_id2
                    relation_type = relation_type_idxs[bid][relation_id]
                    if relation_type > 0:
                        entity1_type = entity_type_idxs[bid][entity_id1]
                        entity2_type = entity_type_idxs[bid][entity_id2]
                        # print('relation type: {}, entity1 type: {}, entity2 type: {}'.format(relation_type,
                        #                                                                      entity1_type,
                        #                                                                      entity2_type))
                        assert entity1_type * entity2_type > 0
                        # connect entity type to the relation type that it participates in
                        # - for entity type 1:
                        # label_adj[entity_type_start + entity1_type][relation_type_start + relation_type] = 1.0

                        label_adj += self.differentiable_matrix_filling(
                            row=entity_type_start + entity1_type,
                            column=relation_type_start + relation_type
                        )

                        # label_adj[relation_type_start + relation_type][entity_type_start + entity1_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=relation_type_start + relation_type,
                            column=entity_type_start + entity1_type
                        )
                        # - for entity type 2:
                        # label_adj[entity_type_start + entity2_type][relation_type_start + relation_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=entity_type_start + entity2_type,
                            column=relation_type_start + relation_type
                        )
                        # label_adj[relation_type_start + relation_type][entity_type_start + entity2_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=relation_type_start + relation_type,
                            column=entity_type_start + entity2_type
                        )
            # capturing occurrence of (event, entity type) and (entity type, entity type) pairs
            for trigger_id in range(max_trigger_num):
                for entity_id in range(max_entity_num):
                    role_id = trigger_id * max_entity_num + entity_id
                    role_type = role_type_idxs[bid][role_id]
                    if role_type > 0:
                        event_type = event_type_idxs[bid][trigger_id]
                        entity_type = entity_type_idxs[bid][entity_id]
                        # print('role type: {}, event type: {}, entity type: {}'.format(
                        #     role_type, event_type, entity_type
                        # ))
                        assert event_type * entity_type > 0
                        # connect entity type to the event type that it serves as an argument
                        # label_adj[entity_type_start + entity_type][event_type_start + event_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=entity_type_start + entity_type,
                            column=event_type_start + event_type
                        )
                        # label_adj[event_type_start + event_type][entity_type_start + entity_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=event_type_start + event_type,
                            column=entity_type_start + entity_type
                        )
                        # connect entity type to the role that it plays in this event
                        # label_adj[entity_type_start + entity_type][role_type_start + role_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=entity_type_start + entity_type,
                            column=role_type_start + role_type
                        )
                        # label_adj[role_type_start + role_type][entity_type_start + entity_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=role_type_start + role_type,
                            column=entity_type_start + entity_type
                        )
                        # connect event type to the role type to capture possible argument types of each event type
                        # label_adj[event_type_start + event_type][role_type_start + role_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=event_type_start + event_type,
                            column=role_type_start + role_type
                        )
                        # label_adj[role_type_start + role_type][event_type_start + event_type] = 1.0
                        label_adj += self.differentiable_matrix_filling(
                            row=role_type_start + role_type,
                            column=event_type_start + event_type
                        )
            all_label_adjs.append(label_adj)
        final_adj = torch.stack(all_label_adjs, dim=0)  # [batch size, num types, num types]
        return final_adj

    def get_pair_reps(self, wgcn_reps, num_a, num_b, sdp_mask, pair_mask):
        batch_size, num_words, word_dim = wgcn_reps.shape
        wgcn_reps = wgcn_reps.repeat(1, num_a * num_b, 1)  # [batch size, (num a * num b) * num words, word dim]
        wgcn_reps = wgcn_reps.view(batch_size * num_a * num_b, num_words, word_dim)
        sdp_reps = max_pooling3d(sdp_mask, wgcn_reps).view(batch_size, -1,
                                                           word_dim)  # [batch size, num a * num b, word dim]
        sdp_reps = sdp_reps * pair_mask
        return sdp_reps

    def embed_nodes(self, node_reps, embed):
        reps = torch.cat(
            [node_reps,
             embed.squeeze().unsqueeze(0).unsqueeze(0).repeat(node_reps.shape[0], node_reps.shape[1],
                                                              1)],
            dim=2
        )  # [batch size, num nodes, word dim + task embed]
        return reps

    def get_task_specific_reps(self, gcn_input, predict):
        '''
        entity_idxs, entity_masks, max_entity_num, max_entity_len,
        trigger_idxs, trigger_masks, max_trigger_num, max_trigger_len
        '''
        input_reps = gcn_input['bert_outputs']

        entity_idxs = gcn_input['graphs_info']['entity_idxs']
        trigger_idxs = gcn_input['graphs_info']['trigger_idxs']

        entity_idxs = input_reps.new_tensor(entity_idxs, dtype=torch.long)
        trigger_idxs = input_reps.new_tensor(trigger_idxs, dtype=torch.long)

        # entity type reprs
        lent_reprs = compute_span_reprs(input_reps, entity_idxs)
        lent_bsz, lent_num, _ = lent_reprs.size()
        lent_dom_labels = lent_reprs.new_zeros(lent_bsz, lent_num).int() + self.dom_emb_type['lent']
        lent_dom_embs   = self.dom_embedding(lent_dom_labels)
        lent_dom_reprs  = self.dom_ffn(lent_dom_embs)
        lent_dom_cond_reprs = self._combine_repr(lent_reprs, lent_dom_reprs)

        # trigger type reprs
        ltrig_reprs = compute_span_reprs(input_reps, trigger_idxs)
        ltrig_bsz, ltrig_num, _ = ltrig_reprs.size()
        ltrig_dom_labels = ltrig_reprs.new_zeros(ltrig_bsz, ltrig_num).int() + self.dom_emb_type['ltrig']
        ltrig_dom_embs   = self.dom_embedding(ltrig_dom_labels)
        ltrig_dom_reprs  = self.dom_ffn(ltrig_dom_embs)
        ltrig_dom_cond_reprs = self._combine_repr(ltrig_reprs, ltrig_dom_reprs)

        ret = {
            'labeled_cond': [lent_dom_cond_reprs, ltrig_dom_cond_reprs],
            'labeled_dom': [lent_dom_reprs, ltrig_dom_reprs],
        }
        if self.config.grda and not predict:
            input_reps = gcn_input['unlabeled']['bert_outputs']

            entity_idxs = gcn_input['unlabeled']['graphs_info'][0]
            trigger_idxs = gcn_input['unlabeled']['graphs_info'][3]

            entity_idxs = input_reps.new_tensor(entity_idxs, dtype=torch.long)
            trigger_idxs = input_reps.new_tensor(trigger_idxs, dtype=torch.long)

            # entity type reprs
            uent_reprs = compute_span_reprs(input_reps, entity_idxs)
            uent_bsz, uent_num, _ = uent_reprs.size()
            uent_dom_labels = uent_reprs.new_zeros(uent_bsz, uent_num).int() + self.dom_emb_type['uent']
            uent_dom_embs   = self.dom_embedding(uent_dom_labels)
            uent_dom_reprs  = self.dom_ffn(uent_dom_embs)
            uent_dom_cond_reprs = self._combine_repr(uent_reprs, uent_dom_reprs)

            # trigger type reprs
            utrig_reprs = compute_span_reprs(input_reps, trigger_idxs)
            utrig_bsz, utrig_num, _ = utrig_reprs.size()
            utrig_dom_labels = utrig_reprs.new_zeros(utrig_bsz, utrig_num).int() + self.dom_emb_type['utrig']
            utrig_dom_embs   = self.dom_embedding(utrig_dom_labels)
            utrig_dom_reprs  = self.dom_ffn(utrig_dom_embs)
            utrig_dom_cond_reprs = self._combine_repr(utrig_reprs, utrig_dom_reprs)

            ret['unlabeled_cond'] = [uent_dom_cond_reprs, utrig_dom_cond_reprs]
            ret['unlabeled_dom'] = [uent_dom_reprs, utrig_dom_reprs]

        # relation type reprs
        e1_reprs, e2_reprs = compute_binary_reprs(lent_reprs, lent_reprs)
        ee_reprs = torch.cat([e1_reprs, e2_reprs, e1_reprs * e2_reprs, torch.abs(e1_reprs - e2_reprs)], dim=2)

        # role type reprs
        t_reprs, e_reprs = compute_binary_reprs(ltrig_reprs, lent_reprs)
        te_reprs = torch.cat([t_reprs, e_reprs, t_reprs * e_reprs, torch.abs(t_reprs - e_reprs)], dim=2)

        # print(f'ltrig_reprs: {ltrig_reprs.size()} | lent_reprs: {lent_reprs.size()}|')
        # print(f'e1_reprs: {e1_reprs.size()} | e2_reprs: {e2_reprs.size()}|')
        # print(f't_reprs: {t_reprs.size()} | e_reprs: {e_reprs.size()}|')
        # print(f'ee_reprs: {ee_reprs.size()} | te_reprs: {te_reprs.size()}|')
        # print()

        te_reprs = self.role_trans(
            te_reprs
        )
        ee_reprs = self.relation_trans(
            ee_reprs
        )

        trigger_reps = self.embed_nodes(ltrig_reprs, self.event_embed)
        entity_reps = self.embed_nodes(lent_reprs, self.entity_embed)
        role_reps = self.embed_nodes(te_reprs, self.role_embed)
        relation_reps = self.embed_nodes(ee_reprs, self.relation_embed)
        node_reps = torch.cat(
            [trigger_reps, entity_reps, role_reps, relation_reps],
            dim=1
        )  # [batch size, num triggers + num entities + num roles + num relations, word dim + task embed]

        return node_reps, ret

    def get_task_adj(self, graph_info):
        '''
        As we always have the nodes in the following order:
        trigger 1 -> trigger 2 -> .... -> trigger T ->
        entity 1 -> entity 2 -> ... -> entity E ->
        (trigger 1, entity 1) -> (trigger 1, entity 2) -> ... -> (trigger T, entity E) ->
        (entity 1, entity 1) -> (entity 1, entity 2) -> ... -> (entity E, entity E)
        We can leverage this fact to construct the graph as follows.
        '''
        with torch.no_grad():
            task_adj_list = []
            batch_size = len(graph_info['trigger_nums'])
            max_trigger_num = graph_info['max_trigger_num']  # max trigger num
            max_entity_num = graph_info['max_entity_num']  # max entity num
            num_nodes = max_trigger_num + max_entity_num + max_trigger_num * max_entity_num + max_entity_num * max_entity_num

            trigger_start = 0
            entity_start = max_trigger_num
            role_start = max_trigger_num + max_entity_num
            relation_start = max_trigger_num + max_entity_num + max_trigger_num * max_entity_num

            for bid in range(batch_size):
                task_adj = [[0] * num_nodes for _ in range(num_nodes)]
                for node_id in range(num_nodes):
                    if node_id < max_trigger_num:  # triggers
                        trigger_id = node_id - trigger_start
                        if trigger_id < graph_info['trigger_nums'][bid]:  # non-padding triggers
                            # connect to related roles
                            for k in range(role_start + trigger_id * max_entity_num,
                                           role_start + (trigger_id + 1) * max_entity_num):
                                task_adj[node_id][k] = 1
                                task_adj[k][node_id] = 1

                    elif node_id < max_trigger_num + max_entity_num:  # entities
                        entity_id = node_id - entity_start
                        if entity_id < graph_info['entity_nums'][bid]:  # non-padding entities
                            # cannot connect to triggers
                            for k in range(max_trigger_num):  # connect to related roles
                                if k < graph_info['trigger_nums'][bid]:
                                    task_adj[node_id][role_start + k * max_entity_num + entity_id] = 1
                                    task_adj[role_start + k * max_entity_num + entity_id][node_id] = 1

                            for k in range(relation_start + entity_id * max_entity_num,
                                           relation_start + entity_id * max_entity_num + graph_info['entity_nums'][
                                               bid]):  # connect to related relations
                                task_adj[node_id][k] = 1
                                task_adj[k][node_id] = 1

                    elif node_id < max_trigger_num + max_entity_num + max_trigger_num * max_entity_num:  # roles
                        role_id = node_id - role_start
                        # connections to triggers are handled in trigger part.

                        '''
                        role ids ------------------------------------------------>
                        0, 		1, 		2, 		3, 		4, 		5, 		6, 		7
                        (0,0)   (0,1)   (0,2)   (1,0)   (1,1)   (1,2)   (2,0)   (2,1)
                        (trigger id, entity id) ----------------------------------->
                        
                        entity_num = 3
                        role id = 7
                        -> trigger id = 7//3 = 2
                        -> entity id = 7%3 = 1
                        '''
                        t_id = role_id // max_entity_num
                        e_id = role_id % max_entity_num
                        if t_id < graph_info['trigger_nums'][bid] and e_id < graph_info['entity_nums'][
                            bid]:  # non-padding relations
                            # connect to other roles that share the same trigger
                            for k in range(role_start + t_id * max_entity_num,
                                           role_start + t_id * max_entity_num + graph_info['entity_nums'][bid]):
                                task_adj[node_id][k] = 1
                                task_adj[k][node_id] = 1
                            # connect to relations that has the same entity involved
                            for k in range(relation_start + e_id * max_entity_num,
                                           relation_start + e_id * max_entity_num + graph_info['entity_nums'][
                                               bid]):  # connect to related relations
                                task_adj[node_id][k] = 1
                                task_adj[k][node_id] = 1
                    else:  # relations
                        relation_id = node_id - relation_start
                        # connections to related roles are handled in role part.
                        '''
                        role ids ------------------------------------------------>
                        0, 		1, 		2, 		3, 		4, 		5, 		6, 		7
                        (0,0)   (0,1)   (0,2)   (1,0)   (1,1)   (1,2)   (2,0)   (2,1)
                        (entity 1 id, entity 2 id) ----------------------------------->
        
                        entity_num = 3
                        role id = 7
                        -> trigger id = 7//3 = 2
                        -> entity id = 7%3 = 1
                        '''
                        entity_1_id = relation_id // max_entity_num
                        entity_2_id = relation_id % max_entity_num
                        if entity_1_id < graph_info['entity_nums'][bid] and entity_2_id < graph_info['entity_nums'][
                            bid]:  # non-padding relations
                            # connect to other relations that share the entity 1
                            for k in range(relation_start + entity_1_id * max_entity_num,
                                           relation_start + entity_1_id * max_entity_num + graph_info['entity_nums'][
                                               bid]):
                                task_adj[node_id][k] = 1
                                task_adj[k][node_id] = 1
                            # connect to other relations that share the entity 2
                            for k in range(relation_start + entity_2_id * max_entity_num,
                                           relation_start + entity_2_id * max_entity_num + graph_info['entity_nums'][
                                               bid]):
                                task_adj[node_id][k] = 1
                                task_adj[k][node_id] = 1
                task_adj_list.append(task_adj)
            return torch.tensor(task_adj_list, requires_grad=False).float().cuda()

    def _combine_repr(self, reprs, dom_reprs):
        #NEW:
        cond_repr = torch.cat([reprs, dom_reprs], dim=2)
        cond_repr = self.combine_ffn(cond_repr)
        return cond_repr