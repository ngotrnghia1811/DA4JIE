from easydict import EasyDict as edict

import torch
import torch.nn

from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

# from transformers import BertModel

from .model_utils import *

from ..modules import Graph, CRF, BertModel, GCN, AGIE_GTN, HierachicalGCN, AGIE_HierachicalGCN
from ..util import *


class AGIE(nn.Module):
    def __init__(self, args, config, vocabs):
        super().__init__()
        self.args = args
        self.config = config
        self.vocabs = vocabs

        #NOTE: vocabularies
        self.entity_label_stoi = vocabs['entity_label']  # BIO tags for [ORG, PER, GPE, ...]
        self.trigger_label_stoi = vocabs['trigger_label']  # BIO tags for [Life:Die, Conflict:Attack, ...]
        self.mention_type_stoi = vocabs['mention_type']  # NAME, PRONOUN, NOMINAL, UNK
        self.entity_type_stoi = vocabs['entity_type']  # [ORG, PER, GPE, ...]
        self.event_type_stoi = vocabs['event_type']  # [Life:Die, Conflict:Attack, ...]
        self.relation_type_stoi = vocabs['relation_type']  # [PART-WHOLE, PER-SOC, ORG-AFF, ...]
        self.role_type_stoi = vocabs['role_type']  # [Victim, Prosecutor, Person, Attacker, ..., O]

        self.entity_label_itos = {i: s for s, i in self.entity_label_stoi.items()}
        self.trigger_label_itos = {i: s for s, i in self.trigger_label_stoi.items()}
        self.entity_type_itos = {i: s for s, i in self.entity_type_stoi.items()}
        self.event_type_itos = {i: s for s, i in self.event_type_stoi.items()}
        self.relation_type_itos = {i: s for s, i in self.relation_type_stoi.items()}
        self.role_type_itos = {i: s for s, i in self.role_type_stoi.items()}

        self.entity_label_num = len(self.entity_label_stoi)
        self.trigger_label_num = len(self.trigger_label_stoi)
        self.mention_type_num = len(self.mention_type_stoi)
        self.entity_type_num = len(self.entity_type_stoi)
        self.event_type_num = len(self.event_type_stoi)
        self.relation_type_num = len(self.relation_type_stoi)
        self.role_type_num = len(self.role_type_stoi)

        #QUES: ?
        self.relation_directional = config.relation_directional
        self.symmetric_relations = config.symmetric_relations
        self.symmetric_relation_idxs = {self.relation_type_stoi[r]
                                        for r in self.symmetric_relations}

        #NOTE: BERT encoder
        self.bert_dim = 1024 if 'large' in config.bert_model_name else 768
        self.num_bert_layers = config.bert_config.num_hidden_layers
        self.extra_bert = config.extra_bert
        self.use_extra_bert = config.use_extra_bert
        if self.use_extra_bert:
            self.bert_dim *= 2
        self.bert = BertModel.from_pretrained(config.bert_model_name,
                                              cache_dir=config.bert_cache_dir,
                                              output_hidden_states=True)
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy

        #NOTE: Hierarchical GCN encoder
        self.args.gcn_dim = self.bert_dim
        self.hierarchical_gcn = AGIE_HierachicalGCN(
            args,
            num_types=self.entity_type_num + self.event_type_num + self.relation_type_num + self.role_type_num,
            config=config,
        )
        self.args.gcn_dim += self.args.task_embed_dim

        #NOTE: CRF
        self.entity_crf = CRF(self.entity_label_stoi, bioes=False)
        self.trigger_crf = CRF(self.trigger_label_stoi, bioes=False)

        #! Add CGNF

        #NOTE: for standard BERT / OneIE
        self.use_entity_type = config.use_entity_type

        linear_bias = config.linear_bias
        linear_dropout = config.linear_dropout

        # self.entity_label_ffn = nn.Linear(self.bert_dim, self.entity_label_num, bias=linear_bias)
        # self.trigger_label_ffn = nn.Linear(self.bert_dim, self.trigger_label_num, bias=linear_bias)

        # _kwargs = {
        #     'dropout_prob' : linear_dropout,
        #     'bias'         : linear_bias,
        #     'activation'   : config.linear_activation
        # }
        # self.entity_type_ffn   = Linears([self.bert_dim, self.entity_type_num], **_kwargs)
        # self.mention_type_ffn  = Linears([self.bert_dim, self.mention_type_num], **_kwargs)
        # self.event_type_ffn    = Linears([self.bert_dim, self.event_type_num], **_kwargs)
        # self.relation_type_ffn = Linears([self.bert_dim, self.relation_type_num], **_kwargs)
        # self.role_type_ffn     = Linears([self.bert_dim, self.role_type_num], **_kwargs)

        #NOTE: for FourIE
        entity_hidden_num = self.args.type_embed_dim
        mention_hidden_num = self.args.type_embed_dim
        event_hidden_num = self.args.type_embed_dim
        relation_hidden_num = self.args.type_embed_dim
        role_hidden_num = self.args.type_embed_dim

        role_input_dim = self.args.gcn_dim + (self.entity_type_num if self.use_entity_type else 0)
        self.entity_label_ffn = nn.Linear(self.bert_dim, self.entity_label_num,
                                          bias=linear_bias)
        self.trigger_label_ffn = nn.Linear(self.bert_dim, self.trigger_label_num,
                                           bias=linear_bias)
        self.entity_type_ffn = Linears([self.args.gcn_dim, entity_hidden_num,
                                        self.entity_type_num],
                                       dropout_prob=linear_dropout,
                                       bias=linear_bias,
                                       activation=config.linear_activation)
        self.mention_type_ffn = Linears([self.args.gcn_dim, mention_hidden_num,
                                         self.mention_type_num],
                                        dropout_prob=linear_dropout,
                                        bias=linear_bias,
                                        activation=config.linear_activation)
        self.event_type_ffn = Linears([self.args.gcn_dim, event_hidden_num,
                                       self.event_type_num],
                                      dropout_prob=linear_dropout,
                                      bias=linear_bias,
                                      activation=config.linear_activation)
        self.relation_type_ffn = Linears([self.args.gcn_dim, relation_hidden_num,
                                          self.relation_type_num],
                                         dropout_prob=linear_dropout,
                                         bias=linear_bias,
                                         activation=config.linear_activation)
        self.role_type_ffn = Linears([role_input_dim, role_hidden_num,
                                      self.role_type_num],
                                     dropout_prob=linear_dropout,
                                     bias=linear_bias,
                                     activation=config.linear_activation)

        #NOTE: Loss Functions
        self.entity_criteria   = torch.nn.CrossEntropyLoss()  # Use for both event + argument
        self.event_criteria    = torch.nn.CrossEntropyLoss()
        self.mention_criteria  = torch.nn.CrossEntropyLoss()
        self.relation_criteria = torch.nn.CrossEntropyLoss()
        self.role_criteria     = torch.nn.CrossEntropyLoss()

        #NOTE: Others
        self.pad_vector = nn.Parameter(torch.randn(1, 1, self.bert_dim))

        #NOTE: FiLM GCN
        self.film_gcn = GCN(
            in_dim=self.bert_dim,
            hidden_dim=self.bert_dim,
            out_dim=1,
            num_layers=config.gcn_config.num_layers, 
            gcn_dropout=self.args.gcn_drop,
            skip_connection=config.gcn_config.skip_connection,
            anneal=config.gcn_config.anneal,
        )

        #NOTE: GTN
        self.gtn = AGIE_GTN(
            num_edge=config.gtn.num_edge,
            num_channels=config.gtn.num_channels, 
            num_layers=config.gtn.num_layers,
            norm=config.gtn.norm,
            w_in=self.bert_dim,
            w_out=self.bert_dim,
            num_bert_layers=self.num_bert_layers,
            device=self.config.device,
        )

        #NOTE: DEP
        self.dep_type_stoi = vocabs['deprel_type']
        self.dep_type_itos = {i: s for s, i in self.dep_type_stoi.items()}
        self.dep_embeddings = nn.Embedding(len(self.dep_type_stoi), self.bert_dim, padding_idx=0)

        #NOTE: GRDA
        if config.grda:
            #NEW: GRDA
            self.grda = GRDA(config)
            self._init_optimizers(config)


    def adaptive_pooling(self, A, N, asgm):
        # print(f'A: {A.size()} | N: {N.size()} | Asgm: {asgm.size()}')
        ret_graph = edict()
        #NEW: FiLM | No beta for now
        alpha = self.film_gcn(N, A)
        # print(alpha.device, asgm.device)
        asgm = alpha.repeat(1,1,asgm.size(-1)) * asgm
        # Pool
        # print(N.size(), asgm.size())
        ret_graph.N = torch.bmm(asgm.permute(0,2,1), N)
        ret_graph.A = torch.bmm(torch.bmm(asgm.permute(0,2,1), A), asgm)
        # print(ret_graph.N.size(), ret_graph.A.size())
        return ret_graph

    def compute_attention_graph(self, all_bert_outputs, batch):
        #NOTE: Adj Matrix Opertation
        token_lens = batch.token_lens
        max_piece_len = batch.piece_idxs.size(1)

        all_attention_probs = all_bert_outputs.attention_probs
        position_embeddings = all_bert_outputs.position_embeddings

        max_token_len = max([len(i) for i in token_lens])
        token_asgm = convert_token_len_to_assignment(token_lens, max_piece_len, max_token_len).to(self.config.device)
        # print(token_asgm)

        attn_graphs = []
        bsz = token_asgm.size(0)
        for layer, adj in enumerate(all_attention_probs):
            #QUES: adj ~ #head x seq x seq ?
            #ANSW: Currently just average. should try gtn later
            A = torch.mean(adj, dim=1)
            N = position_embeddings.repeat(bsz, 1, 1)
            attn_graph = self.adaptive_pooling(A, N, token_asgm)
            attn_graphs.append(attn_graph)
        
        return attn_graphs

    def compute_deprel_graph(self, batch):
        dep_heads = batch.dep_heads
        dep_rels = batch.dep_rels
        dep_graphs = edict()
        dep_graphs.A = convert_dep_head_to_adj(dep_heads).to(self.config.device)
        # print(dep_rels)
        dep_graphs.N = self.dep_embeddings(dep_rels)
        return dep_graphs

    def compute_ctx_inv_graph(self, attn_graphs, dep_graphs):
        #! Only use original Pos_emb for now
        Ns = [attn_graphs[l].N.unsqueeze(-1) for l in range(self.num_bert_layers)]
        Ns.append(dep_graphs.N.unsqueeze(-1))
        #! Just average all node_embs for now
        Ns = torch.mean(torch.cat(Ns, dim=-1), dim=-1)

        As = []
        for l in range(self.num_bert_layers):
            A = torch.cat([attn_graphs[l].A.unsqueeze(-1), dep_graphs.A.unsqueeze(-1)], dim=-1)
            As.append(A)
        
        ci_graph = edict()
        ci_graph.A, ci_graph.N = self.gtn(As, Ns)
        return ci_graph

    def encode(self, piece_idxs, attention_masks, token_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths

        bert_outputs : [batch size, num words, rep dim] 
        #NOTE: Encode token-wise, then 1st/ave for word-wise
        #QUES: Do AGIE still use this or let As Matrix handle it?
        #QUES: CRF work on word or token ?
        """
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        bert_outputs = all_bert_outputs.last_hidden_state

        if self.use_extra_bert:
            raise ValueError("not support extra bert")
            # extra_bert_outputs = all_bert_outputs[2][self.extra_bert]
            # bert_outputs = torch.cat([bert_outputs, extra_bert_outputs], dim=2)

        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets)
            # + 1 because the first vector is for [CLS]
            offsets = offsets + 1
            bert_outputs = compute_word_reprs_first(bert_outputs, offsets)
        elif self.multi_piece == 'average':
            # average all pieces for multi-piece words
            idxs, token_num, token_len = token_lens_to_idxs(token_lens)
            idxs = piece_idxs.new(idxs) + 1
            bert_outputs = compute_word_reps_avg(bert_outputs, idxs)
        else:
            raise ValueError('Unknown multi-piece token handling strategy: {}'
                             .format(self.multi_piece))
        bert_outputs = self.bert_dropout(bert_outputs)
        return bert_outputs, all_bert_outputs

    def identification(self, bert_outputs, batch, unlabeled=False):
        # identification
        entity_label_scores = self.entity_label_ffn(bert_outputs)
        trigger_label_scores = self.trigger_label_ffn(bert_outputs)
        
        entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
        trigger_label_scores = self.trigger_crf.pad_logits(trigger_label_scores)
        
        if unlabeled:
            _, unlabeled_entity_label_preds = self.entity_crf.viterbi_decode(entity_label_scores,
                                                                   batch.token_nums)
            _, unlabeled_trigger_label_preds = self.trigger_crf.viterbi_decode(trigger_label_scores,
                                                                     batch.token_nums)
            unlabeled_entities = tag_paths_to_spans(unlabeled_entity_label_preds,
                                          batch.token_nums,
                                          self.entity_label_stoi)
            unlabeled_triggers = tag_paths_to_spans(unlabeled_trigger_label_preds,
                                          batch.token_nums,
                                          self.trigger_label_stoi)
            return unlabeled_entities, unlabeled_triggers
        else:
            entity_label_loglik = self.entity_crf.loglik(entity_label_scores,
                                                        batch.entity_label_idxs,
                                                        batch.token_nums)
            trigger_label_loglik = self.trigger_crf.loglik(trigger_label_scores,
                                                        batch.trigger_label_idxs,
                                                        batch.token_nums)
            return entity_label_scores, entity_label_loglik, \
                trigger_label_scores, trigger_label_loglik

    def classification(self, bert_outputs, batch, graphs, entity_types_onehot=None,
                       predict=False, ci_graphs=None, unlabeled=None):
        """
        #! Assignment Loss , NAT loss
        return:
            entity_type_scores: 
            mention_type_scores:
            event_type_scores:
            relation_type_scores:
            role_type_scores:
            pattern_loss (remove from FourIE)
        """
        entity_idxs, entity_num, entity_len, \
        trigger_idxs, trigger_num, trigger_len = graphs_to_node_idxs(graphs)
        # print(entity_num, entity_len, trigger_num, trigger_len)

        # print(bert_outputs.size(), entity_num, trigger_num)
        max_token_len = bert_outputs.size(1)
        lent_true_asgm_mat = convert_span_to_assignment(entity_idxs, max_token_len, entity_num).to(self.config.device)
        ltrig_true_asgm_mat = convert_span_to_assignment(trigger_idxs, max_token_len, trigger_num).to(self.config.device)

        lent_ci_pooled  = self.adaptive_pooling(ci_graphs.lA, ci_graphs.lN, lent_true_asgm_mat)
        ltrig_ci_pooled = self.adaptive_pooling(ci_graphs.lA, ci_graphs.lN, ltrig_true_asgm_mat)

        #TODO: Try CGNF
        # entity_type_scores, mention_type_scores, event_type_scores, \
        #  relation_type_scores, role_type_scores = self.cgnf(clss_inputs, predicts)

        # ******* task interaction via hierarchical GCN **********
        gcn_input = {
            'bert_outputs': bert_outputs,
            'entity_type_ffn': self.entity_type_ffn,
            'mention_type_ffn': self.mention_type_ffn,
            'event_type_ffn': self.event_type_ffn,
            'relation_type_ffn': self.relation_type_ffn,
            'role_type_ffn': self.role_type_ffn,
            'entity_type_num': self.entity_type_num,
            'event_type_num': self.event_type_num,
            'relation_type_num': self.relation_type_num,
            'role_type_num': self.role_type_num,
            'batch': batch,
            'graphs_info': {
                'entity_idxs': entity_idxs,
                'max_entity_num': entity_num,
                'entity_len': entity_len,
                'trigger_idxs': trigger_idxs,
                'max_trigger_num': trigger_num,
                'trigger_len': trigger_len,
                'trigger_nums': [len(graph.triggers) for graph in graphs],  # [batch size, ]
                'entity_nums': [len(graph.entities) for graph in graphs]  # [batch size, ]
            },
           'lent_ci_pooled' : lent_ci_pooled,
            'ltrig_ci_pooled': ltrig_ci_pooled,
            }

        if self.config.grda and not predict:
            unlabeled['graphs_info'] = graphs_to_node_idxs(unlabeled['graphs'])
            uentity_idxs,  uentity_num,  uentity_len, \
            utrigger_idxs, utrigger_num, utrigger_len = unlabeled['graphs_info']

            umax_token_len = unlabeled.bert_outputs.size(1)
            uent_true_asgm_mat  = convert_span_to_assignment(uentity_idxs,  umax_token_len, uentity_num).to(self.config.device)
            utrig_true_asgm_mat = convert_span_to_assignment(utrigger_idxs, umax_token_len, utrigger_num).to(self.config.device)

            uent_ci_pooled  = self.adaptive_pooling(ci_graphs.uA, ci_graphs.uN, uent_true_asgm_mat)
            utrig_ci_pooled = self.adaptive_pooling(ci_graphs.uA, ci_graphs.uN, utrig_true_asgm_mat)
            gcn_input.update(
                {
                    'unlabeled': unlabeled,
                    'uent_ci_pooled' : uent_ci_pooled,
                    'utrig_ci_pooled': utrig_ci_pooled,
                }
            )


        entity_type_scores, mention_type_scores, event_type_scores, relation_type_scores, role_type_scores, pattern_loss, dom_cond_reprs = self.hierarchical_gcn(gcn_input, predict)

        return entity_type_scores, mention_type_scores, event_type_scores, relation_type_scores, role_type_scores, pattern_loss, dom_cond_reprs

    def forward(self, batch, unlabeled_batch=None):
        """
        """

        # Bert Encode
        bert_outputs, all_bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()

        # Attn Graph
        lattn_graphs = self.compute_attention_graph(all_bert_outputs, batch)
        # DepRel Graph
        ldep_graphs = self.compute_deprel_graph(batch)
        # Context-Invariant Graph
        lci_graphs = self.compute_ctx_inv_graph(lattn_graphs, ldep_graphs)
        ci_graphs = edict()
        ci_graphs.lA = lci_graphs.A
        ci_graphs.lN = lci_graphs.N

        # Identification
        #! Add CI graph here
        _, entity_label_loglik, _, trigger_label_loglik = self.identification(bert_outputs, batch)

        #NEW: Add unlabeled_bert_outputs
        unlabeled_bert_outputs=None
        unlabeled=None
        if type(unlabeled_batch) != type(None):
            unlabeled_bert_outputs, unlabeled_all_bert_outputs = self.encode(
                                                 unlabeled_batch.piece_idxs,
                                                 unlabeled_batch.attention_masks,
                                                 unlabeled_batch.token_lens)

            #! Add CI graph here
            unlabeled_entities, unlabeled_triggers = self.identification(unlabeled_bert_outputs,
                                                                         unlabeled_batch, unlabeled=True)

            unlabeled_node_graphs = [Graph(e, t, [], [], self.vocabs)
                        for e, t in zip(unlabeled_entities, unlabeled_triggers)]

            # Attn Graph
            uattn_graphs = self.compute_attention_graph(unlabeled_all_bert_outputs, unlabeled_batch)
            # DepRel Graph
            udep_graphs = self.compute_deprel_graph(unlabeled_batch)
            # Context-Invariant Graph
            uci_graphs = self.compute_ctx_inv_graph(uattn_graphs, udep_graphs)
            ci_graphs.uA = uci_graphs.A
            ci_graphs.uN = uci_graphs.N

            unlabeled = edict({
                'bert_outputs': unlabeled_bert_outputs, 
                'batch': unlabeled_batch, 
                'graphs': unlabeled_node_graphs,
            })

        # Classification
        # Convert Entity Type Idx -> OneHot

        entity_type_scores, mention_type_scores, event_type_scores, relation_type_scores, role_type_scores, pattern_loss, dom_cond_reprs = self.classification(
            bert_outputs, batch, batch.graphs, unlabeled=unlabeled, ci_graphs=ci_graphs)

        entity_type_scores = entity_type_scores.view(-1, self.entity_type_num)
        event_type_scores = event_type_scores.view(-1, self.event_type_num)
        relation_type_scores = relation_type_scores.view(-1, self.relation_type_num)
        role_type_scores = role_type_scores.view(-1, self.role_type_num)
        mention_type_scores = mention_type_scores.view(-1, self.mention_type_num)

        # classification_loss = self.entity_criteria(entity_type_scores, batch.entity_type_idxs) + \
        #                       self.event_criteria(event_type_scores, batch.event_type_idxs) + \
        #                       self.relation_criteria(relation_type_scores, batch.relation_type_idxs) + \
        #                       self.role_criteria(role_type_scores, batch.role_type_idxs) + \
        #                       self.mention_criteria(mention_type_scores, batch.mention_type_idxs)

        losses = edict()
        losses.ent_crf = - entity_label_loglik.mean()
        losses.trig_crf = - trigger_label_loglik.mean()

        losses.entity  = self.entity_criteria(entity_type_scores, batch.entity_type_idxs)
        losses.event = self.event_criteria(event_type_scores, batch.event_type_idxs)
        losses.relation = self.relation_criteria(relation_type_scores, batch.relation_type_idxs)
        losses.role = self.role_criteria(role_type_scores, batch.role_type_idxs)
        losses.mention  = self.mention_criteria(mention_type_scores, batch.mention_type_idxs)
        losses.task = losses.entity + losses.event + losses.relation + losses.role + losses.mention + losses.ent_crf + losses.trig_crf
        # task_loss = classification_loss + losses.ent_crf + losses.trig_crf
        losses.pattern = self.args.lamda * pattern_loss
        # losses.task = task_loss 
        # loss = task_loss + self.args.lamda * pattern_loss
        losses.Clss = losses.task + losses.pattern

        # return loss, task_loss, self.args.lamda * pattern_loss
       #NEW: DANN
        if type(unlabeled_batch) != type(None):
            losses.Discr, losses.Gdom, losses.Enc_gan = self.grda(dom_cond_reprs)
            losses.Clss += losses.Enc_gan.squeeze()
            # loss += loss_discr + loss_dom + loss_enc_gan
        # print(losses)
        return losses


    def predict(self, batch):
        """
        """
        self.eval()

        # Bert Encode
        bert_outputs, all_bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()        

        # Identification
        entities, triggers = self.identification(bert_outputs, batch, unlabeled=True)
        

        # Init Graph without rel and type
        node_graphs = [Graph(e, t, [], [], self.vocabs)
                       for e, t in zip(entities, triggers)]

        # Attn Graph
        attn_graphs = self.compute_attention_graph(all_bert_outputs, batch)
        # DepRel Graph
        dep_graphs = self.compute_deprel_graph(batch)
        # Context-Invariant Graph
        ci_graphs = self.compute_ctx_inv_graph(attn_graphs, dep_graphs)
        ci_graphs.lA = ci_graphs.A
        ci_graphs.lN = ci_graphs.N

        entity_type_scores, mention_type_scores, event_type_scores, relation_type_scores, role_type_scores, _, _ = self.classification(
            bert_outputs, batch, node_graphs, predict=True, ci_graphs=ci_graphs)

        batch_graphs = []
        # Decode each sentence in the batch
        for i in range(batch_size):
            seq_entities, seq_triggers = entities[i], triggers[i]
            spans = [(*i, True) for i in seq_entities] + [(*i, False) for i in seq_triggers]

            entity_num, trigger_num = len(seq_entities), len(seq_triggers)
            if entity_num == 0 and trigger_num == 0:
                # skip decoding
                graph = Graph.empty_graph(self.vocabs)
                batch_graphs.append(graph)
                continue
            graph = self.decode(spans,
                                entity_type_scores[i],
                                mention_type_scores[i],
                                event_type_scores[i],
                                relation_type_scores[i],
                                role_type_scores[i])

            batch_graphs.append(graph)

        self.train()
        return batch_graphs

    def decode(self, spans, entity_type_scores, mention_type_scores, event_type_scores,
               relation_type_scores, role_type_scores):
        graph = Graph.empty_graph(self.vocabs)
        max_entity_num = entity_type_scores.shape[0]

        # decode triggers
        # decode entities
        entity_num, trigger_num = 0, 0
        cand_trg_id = -1
        detected_trg_ids = []
 
        cand_ent_id = -1
        detected_ent_ids = []

        for start, end, _, is_entity_node in spans:
            if is_entity_node:
                cand_ent_id += 1
                node_scores = entity_type_scores[cand_ent_id].tolist()
            else:
                cand_trg_id += 1
                node_scores = event_type_scores[cand_trg_id].tolist()

            label = max([(l, s) for l, s in enumerate(node_scores)], key=lambda x: x[1])[0]
            if label > 0:
                if is_entity_node:
                    detected_ent_ids.append(cand_ent_id)
                    graph.add_entity(start, end, label, 0, 0)
                else:
                    detected_trg_ids.append(cand_trg_id)
                    graph.add_trigger(start, end, label, 0, 0)
                if is_entity_node:
                    entity_num += 1
                else:
                    trigger_num += 1
                    
        # decode relations
        for index_1, id1 in enumerate(detected_ent_ids):
            for index_2 in range(index_1):
                rel_id_left = id1 * max_entity_num + detected_ent_ids[index_2] 
                rel_id_right = detected_ent_ids[index_2] * max_entity_num + id1
                l1, s1 = max([(l, s) for l, s in enumerate(relation_type_scores[rel_id_left])], key=lambda x: x[1])
                l2, s2 = max([(l, s) for l, s in enumerate(relation_type_scores[rel_id_right])], key=lambda x: x[1])
                label = l1 if s1 > s2 else l2
                if label > 0:
                    graph.add_relation(index_1, index_2, label, 0, 0)
        # decode roles
        for index_1, id1 in enumerate(detected_trg_ids):
            for index_2, id2 in enumerate(detected_ent_ids):
                role_id = id1 * max_entity_num + id2
                role_scores = role_type_scores[role_id]
                label = max([(l, s) for l, s in enumerate(role_scores)], key=lambda x: x[1])[0]
                if label > 0:
                    graph.add_role(index_1, index_2, label, 0, 0)
        # predict mention types
        _, mention_types = mention_type_scores.max(dim=1)
        mention_types = [mt for k, mt in enumerate(mention_types) if k in detected_ent_ids]
        mention_list = [(i, j, l.item()) for (i, j, k), l
                        in zip(graph.entities, mention_types)]
        graph.mentions = mention_list

        return graph

    def _init_optimizers(self, config):
        self.optimizers = edict()
        self.lr_schedulers = edict()

        Clss_param_groups = [
        { # BERT
            'params': [p for n, p in self.named_parameters() if n.startswith('bert')],
            'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
        },
        { # ffn for dwstream tasks````
            'params': [p for n, p in self.named_parameters() if not n.startswith('bert')
                    and 'crf' not in n and 'global_feature' not in n and 'grda' not in n and 'gtn' not in n and 'film' not in n],
            'lr': config.learning_rate, 'weight_decay': config.weight_decay
        },
        { # CRF and global_feat
            'params': [p for n, p in self.named_parameters() if not n.startswith('bert')
                    and ('crf' in n or 'global_feature' in n)],
            'lr': config.learning_rate, 'weight_decay': 0
        },
        { # GTN
            'params': [p for n, p in self.named_parameters() if not n.startswith('bert')
                    and ('gtn' in n)],
            'lr': config.gtn_learning_rate, 'weight_decay': 0
        },
        { # FILM_GCN
            'params': [p for n, p in self.named_parameters() if not n.startswith('bert')
                    and ('film_gcn' in n)],
            'lr': config.film_learning_rate, 'weight_decay': 0
        },
        ]


        #NEW: GRDA
        Discr_param_groups = [
        { # Discr
            'params': [p for n, p in self.named_parameters() if n.startswith('grda') and 'Discr' in n],
            'lr': config.grda.Discr_lr, 
            'weight_decay':config.grda.Discr_weight_decay
        },
        ]
        Gdom_param_groups = [
        { # Gdom
            'params': [p for n, p in self.named_parameters() if n.startswith('grda') and 'Gdom' in n],
            'lr': config.grda.Gdom_lr, 
            'weight_decay':config.grda.Gdom_weight_decay
        },
        ]

        batch_num = config.batch_num
        self.optimizers.Clss = AdamW(params=Clss_param_groups)
        self.lr_schedulers.Clss = get_linear_schedule_with_warmup(
                                           self.optimizers.Clss,
                                           num_warmup_steps=batch_num * 5,
                                           num_training_steps=batch_num * 100)

        self.optimizers.Discr = AdamW(params=Discr_param_groups)
        self.lr_schedulers.Discr = get_linear_schedule_with_warmup(
                                           self.optimizers.Discr,
                                           num_warmup_steps=batch_num * 5,
                                           num_training_steps=batch_num * 100)

        self.optimizers.Gdom = AdamW(params=Gdom_param_groups)
        self.lr_schedulers.Gdom = get_linear_schedule_with_warmup(
                                           self.optimizers.Gdom,
                                           num_warmup_steps=batch_num * 5,
                                           num_training_steps=batch_num * 100)

    def alt_optim_step(self, losses):
        #QUES: Currently, both step use the same reprs. Recompute Needed ?
        #ANSW:
        #NEW: retain_graph=True so bkwd doesn't free computation graph
        # Update Graph_dom enc
        self.optimizers.Gdom.zero_grad()
        losses.Gdom.backward(retain_graph=True)
        # Update Discriminator
        self.optimizers.Discr.zero_grad()
        losses.Discr.backward(retain_graph=True)
        # Update Classification (Encoder + Predictor)
        self.optimizers.Clss.zero_grad()
        losses.Clss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.parameters(), self.config.grad_clipping)

        # self.optimizers.Gdom.step()
        # self.optimizers.Discr.step()
        # self.optimizers.Clss.step()

        # learning rate decay
        for optimizer, lr_scheduler in zip(self.optimizers.values(),
                                           self.lr_schedulers.values()):
            optimizer.step()
            lr_scheduler.step()
        
