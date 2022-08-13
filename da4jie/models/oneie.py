from easydict import EasyDict as edict

import torch
import torch.nn as nn

from transformers import BertModel, get_linear_schedule_with_warmup, AdamW
from .model_utils import *

from ..modules import Graph, CRF
from ..modules.global_feature import generate_global_feature_vector, generate_global_feature_maps
from ..util import normalize_score, generate_pairwise_idxs, compute_word_reps_avg

class OneIE(nn.Module):
    def __init__(self,
                 config,
                 vocabs,
                 valid_patterns=None):
        super().__init__()
        self.config = config

        # vocabularies
        self.vocabs = vocabs
        self.entity_label_stoi = vocabs['entity_label']
        self.trigger_label_stoi = vocabs['trigger_label']
        self.mention_type_stoi = vocabs['mention_type']
        self.entity_type_stoi = vocabs['entity_type']
        self.event_type_stoi = vocabs['event_type']
        self.relation_type_stoi = vocabs['relation_type']
        self.role_type_stoi = vocabs['role_type']
        self.entity_label_itos = {i:s for s, i in self.entity_label_stoi.items()}
        self.trigger_label_itos = {i:s for s, i in self.trigger_label_stoi.items()}
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
        self.valid_relation_entity = set()
        self.valid_event_role = set()
        self.valid_role_entity = set()
        if valid_patterns:
            self.valid_event_role = valid_patterns['event_role']
            self.valid_relation_entity = valid_patterns['relation_entity']
            self.valid_role_entity = valid_patterns['role_entity']
        self.relation_directional = config.relation_directional
        self.symmetric_relations = config.symmetric_relations
        self.symmetric_relation_idxs = {self.relation_type_stoi[r]
                                        for r in self.symmetric_relations}

        # BERT encoder
        bert_config = config.bert_config
        bert_config.output_hidden_states = True
        self.bert_dim = bert_config.hidden_size
        self.extra_bert = config.extra_bert
        self.use_extra_bert = config.use_extra_bert
        if self.use_extra_bert:
            self.bert_dim *= 2
        self.bert = BertModel.from_pretrained(config.bert_model_name,
                                              cache_dir=config.bert_cache_dir,
                                              output_hidden_states=True)
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy

        # local classifiers
        self.use_entity_type = config.use_entity_type
        self.binary_dim = self.bert_dim * 2
        linear_bias = config.linear_bias
        linear_dropout = config.linear_dropout
        entity_hidden_num = config.entity_hidden_num
        mention_hidden_num = config.mention_hidden_num
        event_hidden_num = config.event_hidden_num
        relation_hidden_num = config.relation_hidden_num
        role_hidden_num = config.role_hidden_num
        role_input_dim = self.binary_dim + (self.entity_type_num if self.use_entity_type else 0)
        self.entity_label_ffn = nn.Linear(self.bert_dim, self.entity_label_num,
                                        bias=linear_bias)
        self.trigger_label_ffn = nn.Linear(self.bert_dim, self.trigger_label_num,
                                         bias=linear_bias)
        self.entity_type_ffn = Linears([self.bert_dim, entity_hidden_num,
                                        self.entity_type_num],
                                       dropout_prob=linear_dropout,
                                       bias=linear_bias,
                                       activation=config.linear_activation)
        self.mention_type_ffn = Linears([self.bert_dim, mention_hidden_num,
                                         self.mention_type_num],
                                        dropout_prob=linear_dropout,
                                        bias=linear_bias,
                                        activation=config.linear_activation)
        self.event_type_ffn = Linears([self.bert_dim, event_hidden_num,
                                       self.event_type_num],
                                      dropout_prob=linear_dropout,
                                      bias=linear_bias,
                                      activation=config.linear_activation)
        self.relation_type_ffn = Linears([self.binary_dim, relation_hidden_num,
                                          self.relation_type_num],
                                         dropout_prob=linear_dropout,
                                         bias=linear_bias,
                                         activation=config.linear_activation)
        self.role_type_ffn = Linears([role_input_dim, role_hidden_num,
                                      self.role_type_num],
                                     dropout_prob=linear_dropout,
                                     bias=linear_bias,
                                     activation=config.linear_activation)
        # global features
        self.use_global_features = config.use_global_features
        self.global_features = config.global_features
        self.global_feature_maps = generate_global_feature_maps(vocabs, valid_patterns)
        self.global_feature_num = sum(len(m) for k, m in self.global_feature_maps.items()
                                      if k in self.global_features or
                                      not self.global_features)
        self.global_feature_weights = nn.Parameter(
            torch.zeros(self.global_feature_num).fill_(-0.0001))
        # decoder
        self.beam_size = config.beam_size
        self.beta_v = config.beta_v
        self.beta_e = config.beta_e
        # loss functions
        self.entity_criteria = torch.nn.CrossEntropyLoss()
        self.event_criteria = torch.nn.CrossEntropyLoss()
        self.mention_criteria = torch.nn.CrossEntropyLoss()
        self.relation_criteria = torch.nn.CrossEntropyLoss()
        self.role_criteria = torch.nn.CrossEntropyLoss()
        # others
        self.entity_crf = CRF(self.entity_label_stoi, bioes=False)
        self.trigger_crf = CRF(self.trigger_label_stoi, bioes=False)
        self.pad_vector = nn.Parameter(torch.randn(1, 1, self.bert_dim))

        #NEW: dom embedding
        self.dom_emb_type = config.dom_emb_type
        self.dom_embedding = nn.Embedding(len(config.dom_emb_type), 
                                          config.dom_emb_size[0])
        self.dom_ffn = Linears(config.dom_emb_size,
                               dropout_prob=linear_dropout,
                               bias=linear_bias,
                               activation=config.linear_activation)
        #NEW: Currently concat [repr, dom]
        self.combine_ffn = Linears([config.dom_emb_size[-1] + self.bert_dim, self.bert_dim],
                                   dropout_prob=linear_dropout,
                                   bias=linear_bias,
                                   activation=config.linear_activation)

        #NEW: GRDA
        self.grda = GRDA(config)

    # def load_bert(self, name, cache_dir=None):
    #     """Load the pre-trained BERT model (used in training phrase)
    #     :param name (str): pre-trained BERT model name
    #     :param cache_dir (str): path to the BERT cache directory
    #     """
    #     print('Loading pre-trained BERT model {}'.format(name))
    #     self.bert = BertModel.from_pretrained(name,
    #                                           cache_dir=cache_dir,
    #                                           output_hidden_states=True)
        self._init_optimizers(config)

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
                    and 'crf' not in n and 'global_feature' not in n],
            'lr': config.learning_rate, 'weight_decay': config.weight_decay
        },
        { # CRF and global_feat
            'params': [p for n, p in self.named_parameters() if not n.startswith('bert')
                    and ('crf' in n or 'global_feature' in n)],
            'lr': config.learning_rate, 'weight_decay': 0
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

    def encode(self, piece_idxs, attention_masks, token_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        bert_outputs = all_bert_outputs[0]

        if self.use_extra_bert:
            extra_bert_outputs = all_bert_outputs[2][self.extra_bert]
            bert_outputs = torch.cat([bert_outputs, extra_bert_outputs], dim=2)

        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets)
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':
            # ONEIE: average all pieces for multi-piece words
            # idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
            # idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            # masks = bert_outputs.new(masks).unsqueeze(-1)
            # bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
            # bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
            # bert_outputs = bert_outputs.sum(2)

            # FOURIE: average all pieces for multi-piece words
            idxs, token_num, token_len = token_lens_to_idxs(token_lens)
            idxs = piece_idxs.new(idxs) + 1
            bert_outputs = compute_word_reps_avg(bert_outputs, idxs)
        else:
            raise ValueError('Unknown multi-piece token handling strategy: {}'
                             .format(self.multi_piece))
        bert_outputs = self.bert_dropout(bert_outputs)
        return bert_outputs

    def scores(self, reprs,
               entity_idxs, entity_types_onehot=None,
               predict=False):
        entity_reprs, trigger_reprs = reprs
        entity_type_scores = self.entity_type_ffn(entity_reprs)
        mention_type_scores = self.mention_type_ffn(entity_reprs)
        event_type_scores = self.event_type_ffn(trigger_reprs)


        batch_size, entity_num, bert_dim = entity_reprs.size()
        batch_size, trigger_num, bert_dim = trigger_reprs.size()

        # relation type score
        ee_idxs = generate_pairwise_idxs(entity_num, entity_num)
        ee_idxs = entity_idxs.new(ee_idxs)
        ee_idxs = ee_idxs.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, bert_dim)
        ee_reprs = torch.cat([entity_reprs, entity_reprs], dim=1)
        ee_reprs = torch.gather(ee_reprs, 1, ee_idxs)
        ee_reprs = ee_reprs.view(batch_size, -1, 2 * bert_dim)
        relation_type_scores = self.relation_type_ffn(ee_reprs)
        
        # role type score
        te_idxs = generate_pairwise_idxs(trigger_num, entity_num)
        te_idxs = entity_idxs.new(te_idxs)
        te_idxs = te_idxs.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, bert_dim)
        te_reprs = torch.cat([trigger_reprs, entity_reprs], dim=1)
        te_reprs = torch.gather(te_reprs, 1, te_idxs)
        te_reprs = te_reprs.view(batch_size, -1, 2 * bert_dim)
        
        if self.use_entity_type:
            if predict:
                entity_type_scores_softmax = entity_type_scores.softmax(dim=2)
                entity_type_scores_softmax = entity_type_scores_softmax.repeat(1, trigger_num, 1)
                te_reprs = torch.cat([te_reprs, entity_type_scores_softmax], dim=2)
            else:
                #NEW: this use true label
                entity_types_onehot = entity_types_onehot.repeat(1, trigger_num, 1)
                te_reprs = torch.cat([te_reprs, entity_types_onehot], dim=2)
        role_type_scores = self.role_type_ffn(te_reprs)

        return (entity_type_scores, mention_type_scores, event_type_scores,
                relation_type_scores, role_type_scores)

    def create_reprs(self, bert_outputs, graphs):
        (
            entity_idxs, entity_masks, entity_num, entity_len,
            trigger_idxs, trigger_masks, trigger_num, trigger_len,
        ) = graphs_to_node_idxs_with_mask(graphs)

        batch_size, _, bert_dim = bert_outputs.size()

        entity_idxs = bert_outputs.new_tensor(entity_idxs, dtype=torch.long)
        trigger_idxs = bert_outputs.new_tensor(trigger_idxs, dtype=torch.long)
        entity_masks = bert_outputs.new_tensor(entity_masks)
        trigger_masks = bert_outputs.new_tensor(trigger_masks)

        # entity type scores
        entity_idxs = entity_idxs.unsqueeze(-1).expand(-1, -1, bert_dim)
        entity_masks = entity_masks.unsqueeze(-1).expand(-1, -1, bert_dim)
        entity_words = torch.gather(bert_outputs, 1, entity_idxs)
        entity_words = entity_words * entity_masks
        entity_words = entity_words.view(batch_size, entity_num, entity_len, bert_dim)
        #NOTE: Arg representation
        entity_reprs = entity_words.sum(2)
        # entity_type_scores = self.entity_type_ffn(entity_reprs)

        # mention type scores
        # mention_type_scores = self.mention_type_ffn(entity_reprs)

        # trigger type scores
        trigger_idxs = trigger_idxs.unsqueeze(-1).expand(-1, -1, bert_dim)
        trigger_masks = trigger_masks.unsqueeze(-1).expand(-1, -1, bert_dim)
        trigger_words = torch.gather(bert_outputs, 1, trigger_idxs)
        trigger_words = trigger_words * trigger_masks
        trigger_words = trigger_words.view(batch_size, trigger_num, trigger_len, bert_dim)
        #NOTE: Trig representation
        trigger_reprs = trigger_words.sum(2)
        # event_type_scores = self.event_type_ffn(trigger_reprs)

        # relation type score
        ee_idxs = generate_pairwise_idxs(entity_num, entity_num)
        ee_idxs = entity_idxs.new(ee_idxs)
        ee_idxs = ee_idxs.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, bert_dim)
        ee_reprs = torch.cat([entity_reprs, entity_reprs], dim=1)
        ee_reprs = torch.gather(ee_reprs, 1, ee_idxs)
        ee_reprs = ee_reprs.view(batch_size, -1, 2 * bert_dim)
        #NOTE: Arg-Arg representation
        # relation_type_scores = self.relation_type_ffn(ee_reprs)

        # # role type score
        # te_idxs = generate_pairwise_idxs(trigger_num, entity_num)
        # te_idxs = entity_idxs.new(te_idxs)
        # te_idxs = te_idxs.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, bert_dim)
        # te_reprs = torch.cat([trigger_reprs, entity_reprs], dim=1)
        # te_reprs = torch.gather(te_reprs, 1, te_idxs)
        # te_reprs = te_reprs.view(batch_size, -1, 2 * bert_dim)

        # #NOTE: Arg-Trig representation
        # if self.use_entity_type:
        #     if predict:
        #         entity_type_scores_softmax = entity_type_scores.softmax(dim=2)
        #         entity_type_scores_softmax = entity_type_scores_softmax.repeat(1, trigger_num, 1)
        #         te_reprs = torch.cat([te_reprs, entity_type_scores_softmax], dim=2)
        #     else:
        #         entity_types_onehot = entity_types_onehot.repeat(1, trigger_num, 1)
        #         te_reprs = torch.cat([te_reprs, entity_types_onehot], dim=2)
        # role_type_scores = self.role_type_ffn(te_reprs)

        # ret = {
        #     'repr': (entity_reprs, entity_reprs, trigger_reprs,
        #         ee_reprs, te_reprs),
        #     'scores': (entity_type_scores, mention_type_scores, event_type_scores,
        #         relation_type_scores, role_type_scores)
        # }

        # return entity_idxs for self.scores
        return (entity_reprs, trigger_reprs, ee_reprs), entity_idxs

    def forward(self, batch, unlabeled_batch=None):
        # encoding
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        #NEW: Add unlabeled_bert_outputs
        unlabeled_bert_outputs=None
        if type(unlabeled_batch) != type(None):
            unlabeled_bert_outputs = self.encode(unlabeled_batch.piece_idxs,
                                                 unlabeled_batch.attention_masks,
                                                 unlabeled_batch.token_lens)

            # identification
            unlabeled_entity_label_scores = self.entity_label_ffn(unlabeled_bert_outputs)
            unlabeled_entity_label_scores = self.entity_crf.pad_logits(unlabeled_entity_label_scores)
            unlabeled_trigger_label_scores = self.trigger_label_ffn(unlabeled_bert_outputs)
            unlabeled_trigger_label_scores = self.trigger_crf.pad_logits(unlabeled_trigger_label_scores)

            _, unlabeled_entity_label_preds = self.entity_crf.viterbi_decode(unlabeled_entity_label_scores,
                                                                   unlabeled_batch.token_nums)
            _, unlabeled_trigger_label_preds = self.trigger_crf.viterbi_decode(unlabeled_trigger_label_scores,
                                                                     unlabeled_batch.token_nums)
            unlabeled_entities = tag_paths_to_spans(unlabeled_entity_label_preds,
                                          unlabeled_batch.token_nums,
                                          self.entity_label_stoi)
            unlabeled_triggers = tag_paths_to_spans(unlabeled_trigger_label_preds,
                                          unlabeled_batch.token_nums,
                                          self.trigger_label_stoi)
            # Graph(entities, triggers, relations, roles, vocabs, mentions=None)
            # print(unlabeled_bert_outputs.size())
            # print(unlabeled_entity_label_preds)
            # print(unlabeled_trigger_label_preds)
            # print(unlabeled_entities)
            # print(unlabeled_triggers)
            unlabeled_node_graphs = [Graph(e, t, [], [], self.vocabs)
                        for e, t in zip(unlabeled_entities, unlabeled_triggers)]
            # unlabeled_scores = self.scores(unlabeled_bert_outputs, unlabeled_node_graphs, unlabeled=True)
            unlabeled_reprs, _ = self.create_reprs(unlabeled_bert_outputs, unlabeled_node_graphs)
            # print(unlabeled_reprs[0].size(), unlabeled_reprs[1].size())
    
            # unlabeled_graphs = self.predict(unlabeled_batch, unlabeled=True, beam_size=1)
            # if self.config.ignore_first_header:
            #     for inst_idx, sent_id in enumerate(batch.sent_ids):
            #         if int(sent_id.split('-')[-1]) < 4:
            #             unlabeled_graphs[inst_idx] = Graph.empty_graph(self.vocabs)
            # for graph in unlabeled_graphs:
            #     graph.clean(relation_directional=self.config.relation_directional,
            #                 symmetric_relations=self.config.symmetric_relations)
            # unlabeled_reprs, _ = self.create_reprs(
            #                     unlabeled_bert_outputs, unlabeled_graphs)
        else:
            unlabeled_reprs = None

        batch_size, _, _ = bert_outputs.size()

        # identification
        entity_label_scores = self.entity_label_ffn(bert_outputs)
        trigger_label_scores = self.trigger_label_ffn(bert_outputs)

        entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
        entity_label_loglik = self.entity_crf.loglik(entity_label_scores,
                                                     batch.entity_label_idxs,
                                                     batch.token_nums)
        trigger_label_scores = self.trigger_crf.pad_logits(trigger_label_scores)
        trigger_label_loglik = self.trigger_crf.loglik(trigger_label_scores,
                                                       batch.trigger_label_idxs,
                                                       batch.token_nums)
        #NEW:
        labeled_reprs, labeled_entity_idxs = self.create_reprs(bert_outputs, batch.graphs)
        
        dom_cond_reprs = self.create_dom_repr(labeled_reprs, unlabeled_reprs)

        # classification
        #NEW: Teacher Forcing - using true labels
        # convert true entity type indices -> one hot
        entity_types = batch.entity_type_idxs.view(batch_size, -1)  # bsz * max_num_arg
        entity_types = torch.clamp(entity_types, min=0)
        entity_types_onehot = bert_outputs.new_zeros(*entity_types.size(),
                                                      self.entity_type_num) 
        entity_types_onehot.scatter_(2, entity_types.unsqueeze(-1), 1) # bsz * max_num_arg * num_ent_type (1-hot)

        # scores = self.scores(bert_outputs, batch.graphs, entity_types_onehot,
                            #  unlabeled_bert_outputs=unlabeled_bert_outputs)
        scores = self.scores(dom_cond_reprs['labeled_cond'], labeled_entity_idxs, entity_types_onehot,)
        (
            entity_type_scores, mention_type_scores, event_type_scores,
            relation_type_scores, role_type_scores
        ) = scores
        entity_type_scores = entity_type_scores.view(-1, self.entity_type_num)
        event_type_scores = event_type_scores.view(-1, self.event_type_num)
        relation_type_scores = relation_type_scores.view(-1, self.relation_type_num)
        role_type_scores = role_type_scores.view(-1, self.role_type_num)
        mention_type_scores = mention_type_scores.view(-1, self.mention_type_num)
        classification_loss = self.entity_criteria(entity_type_scores,
                                                   batch.entity_type_idxs) + \
                              self.event_criteria(event_type_scores,
                                                  batch.event_type_idxs) + \
                              self.relation_criteria(relation_type_scores,
                                                     batch.relation_type_idxs) + \
                              self.role_criteria(role_type_scores,
                                                 batch.role_type_idxs) + \
                              self.mention_criteria(mention_type_scores,
                                                    batch.mention_type_idxs)

        losses = edict()
        loss = classification_loss - entity_label_loglik.mean() - trigger_label_loglik.mean()

        # global features
        if self.use_global_features:
            gold_scores = self.compute_graph_scores(batch.graphs, scores)
            top_graphs = self.generate_locally_top_graphs(batch.graphs, scores)
            top_scores = self.compute_graph_scores(top_graphs, scores)
            global_loss = (top_scores - gold_scores).clamp(min=0)
            loss = loss + global_loss.mean()
        losses.Clss = loss

        #NEW: DANN
        if type(unlabeled_batch) != type(None):
            losses.Discr, losses.Gdom, losses.Enc_gan = self.grda(dom_cond_reprs)
            losses.Clss += losses.Enc_gan.squeeze()
            # loss += loss_discr + loss_dom + loss_enc_gan
        return losses

    def _combine_repr(self, reprs, dom_reprs):
        #NEW:
        cond_repr = torch.cat([reprs, dom_reprs], dim=2)
        cond_repr = self.combine_ffn(cond_repr)
        return cond_repr

    def create_dom_repr(self, labeled_repr, unlabeled_repr=None):
        #NEW: Create Reprs
        # source entity_reprs
        lent_reprs = labeled_repr[0]
        lent_bsz, lent_num, _ = lent_reprs.size()
        lent_dom_labels = lent_reprs.new_zeros(lent_bsz, lent_num).int() + self.dom_emb_type['lent']
        lent_dom_embs   = self.dom_embedding(lent_dom_labels)
        lent_dom_reprs  = self.dom_ffn(lent_dom_embs)
        lent_dom_cond_reprs = self._combine_repr(lent_reprs, lent_dom_reprs)

        # source trigger_reprs
        ltrig_reprs = labeled_repr[1]
        ltrig_bsz, ltrig_num, _ = ltrig_reprs.size()
        ltrig_dom_labels = ltrig_reprs.new_zeros(ltrig_bsz, ltrig_num).int() + self.dom_emb_type['ltrig']
        ltrig_dom_embs   = self.dom_embedding(ltrig_dom_labels)
        ltrig_dom_reprs  = self.dom_ffn(ltrig_dom_embs)
        ltrig_dom_cond_reprs = self._combine_repr(ltrig_reprs, ltrig_dom_reprs)

        ret = {
            'labeled_cond': [lent_dom_cond_reprs, ltrig_dom_cond_reprs],
            'labeled_dom': [lent_dom_reprs, ltrig_dom_reprs],
        }

        if type(unlabeled_repr) != type(None):
            # target entity_reprs
            uent_reprs = unlabeled_repr[0]
            uent_bsz, uent_num, _ = uent_reprs.size()
            uent_dom_labels = uent_reprs.new_zeros(uent_bsz, uent_num).int() + self.dom_emb_type['uent']
            uent_dom_embs   = self.dom_embedding(uent_dom_labels)
            uent_dom_reprs  = self.dom_ffn(uent_dom_embs)
            uent_dom_cond_reprs = self._combine_repr(uent_reprs, uent_dom_reprs)

            # target trigger_reprs
            utrig_reprs = unlabeled_repr[1]
            utrig_bsz, utrig_num, _ = utrig_reprs.size()
            utrig_dom_labels = utrig_reprs.new_zeros(utrig_bsz, utrig_num).int() + self.dom_emb_type['utrig']
            utrig_dom_embs   = self.dom_embedding(utrig_dom_labels)
            utrig_dom_reprs  = self.dom_ffn(utrig_dom_embs)
            utrig_dom_cond_reprs = self._combine_repr(utrig_reprs, utrig_dom_reprs)

            ret['unlabeled_cond'] = [uent_dom_cond_reprs, utrig_dom_cond_reprs]
            ret['unlabeled_dom'] = [uent_dom_reprs, utrig_dom_reprs]
        return ret

    def predict(self, batch, unlabeled=False, beam_size=None):
        if not unlabeled:
            self.eval()

        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()

        # identification
        entity_label_scores = self.entity_label_ffn(bert_outputs)
        entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
        trigger_label_scores = self.trigger_label_ffn(bert_outputs)
        trigger_label_scores = self.trigger_crf.pad_logits(trigger_label_scores)
        _, entity_label_preds = self.entity_crf.viterbi_decode(entity_label_scores,
                                                               batch.token_nums)
        _, trigger_label_preds = self.trigger_crf.viterbi_decode(trigger_label_scores,
                                                                 batch.token_nums)
        entities = tag_paths_to_spans(entity_label_preds,
                                      batch.token_nums,
                                      self.entity_label_stoi)
        triggers = tag_paths_to_spans(trigger_label_preds,
                                      batch.token_nums,
                                      self.trigger_label_stoi)
        # Graph(entities, triggers, relations, roles, vocabs, mentions=None)
        node_graphs = [Graph(e, t, [], [], self.vocabs)
                       for e, t in zip(entities, triggers)]

        labeled_reprs, labeled_entity_idxs = self.create_reprs(bert_outputs, node_graphs)
        labeled_dom_cond_reprs = self.create_dom_repr(labeled_reprs)['labeled_cond']
        # scores = self.scores(bert_outputs, node_graphs, predict=True)
        scores = self.scores(labeled_dom_cond_reprs, labeled_entity_idxs, predict=True)

        max_entity_num = max(max(len(seq_entities) for seq_entities in entities), 1)

        batch_graphs = []
        # Decode each sentence in the batch
        for i in range(batch_size):
            seq_entities, seq_triggers = entities[i], triggers[i]
            spans = sorted([(*i, True) for i in seq_entities] +
                           [(*i, False) for i in seq_triggers],
                           key=lambda x: (x[0], x[1], not x[-1]))
            entity_num, trigger_num = len(seq_entities), len(seq_triggers)
            if entity_num == 0 and trigger_num == 0:
                # skip decoding
                batch_graphs.append(Graph.empty_graph(self.vocabs))
                continue
            graph = self.decode(spans,
                                entity_type_scores=scores[0][i],
                                mention_type_scores=scores[1][i],
                                event_type_scores=scores[2][i],
                                relation_type_scores=scores[3][i],
                                role_type_scores=scores[4][i],
                                entity_num=max_entity_num,
                                beam_size=beam_size)
            batch_graphs.append(graph)

        self.train()
        return batch_graphs

    def compute_graph_scores(self, graphs, scores):
        (
            entity_type_scores, _mention_type_scores,
            trigger_type_scores, relation_type_scores,
            role_type_scores
        ) = scores
        label_idxs = graphs_to_label_idxs(graphs)
        label_idxs = [entity_type_scores.new_tensor(idx,
                                               dtype=torch.long if i % 2 == 0
                                               else torch.float)
                      for i, idx in enumerate(label_idxs)]
        (
            entity_idxs, entity_mask, trigger_idxs, trigger_mask,
            relation_idxs, relation_mask, role_idxs, role_mask
        ) = label_idxs
        # Entity score
        entity_idxs = entity_idxs.unsqueeze(-1)
        entity_scores = torch.gather(entity_type_scores, 2, entity_idxs)
        entity_scores = entity_scores.squeeze(-1) * entity_mask
        entity_score = entity_scores.sum(1)
        # Trigger score
        trigger_idxs = trigger_idxs.unsqueeze(-1)
        trigger_scores = torch.gather(trigger_type_scores, 2, trigger_idxs)
        trigger_scores = trigger_scores.squeeze(-1) * trigger_mask
        trigger_score = trigger_scores.sum(1)
        # Relation score
        relation_idxs = relation_idxs.unsqueeze(-1)
        relation_scores = torch.gather(relation_type_scores, 2, relation_idxs)
        relation_scores = relation_scores.squeeze(-1) * relation_mask
        relation_score = relation_scores.sum(1)
        # Role score
        role_idxs = role_idxs.unsqueeze(-1)
        role_scores = torch.gather(role_type_scores, 2, role_idxs)
        role_scores = role_scores.squeeze(-1) * role_mask
        role_score = role_scores.sum(1)

        score = entity_score + trigger_score + role_score + relation_score

        global_vectors = [generate_global_feature_vector(g, self.global_feature_maps, features=self.global_features)
                          for g in graphs]
        global_vectors = entity_scores.new_tensor(global_vectors)
        global_weights = self.global_feature_weights.unsqueeze(0).expand_as(global_vectors)
        global_score = (global_vectors * global_weights).sum(1)
        score = score + global_score

        return score

    def generate_locally_top_graphs(self, graphs, scores):
        (
            entity_type_scores, _mention_type_scores,
            trigger_type_scores, relation_type_scores,
            role_type_scores
        ) = scores
        max_entity_num = max(max([g.entity_num for g in graphs]), 1)
        top_graphs = []
        for graph_idx, graph in enumerate(graphs):
            entity_num = graph.entity_num
            trigger_num = graph.trigger_num
            _, top_entities = entity_type_scores[graph_idx].max(1)
            top_entities = top_entities.tolist()[:entity_num]
            top_entities = [(i, j, k) for (i, j, _), k in
                            zip(graph.entities, top_entities)]
            _, top_triggers = trigger_type_scores[graph_idx].max(1)
            top_triggers = top_triggers.tolist()[:trigger_num]
            top_triggers = [(i, j, k) for (i, j, _), k in
                            zip(graph.triggers, top_triggers)]
            
            top_relation_scores, top_relation_labels = relation_type_scores[graph_idx].max(1)
            top_relation_scores = top_relation_scores.tolist()
            top_relation_labels = top_relation_labels.tolist()
            top_relations = [(i, j) for i, j in zip(top_relation_scores, top_relation_labels)]
            top_relation_list = []
            for i in range(entity_num):
                for j in range(entity_num):
                    if i < j:
                        score_1, label_1 = top_relations[i * max_entity_num + j]
                        score_2, label_2 = top_relations[j * max_entity_num + i]
                        if score_1 > score_2 and label_1 != 0:
                            top_relation_list.append((i, j, label_1))
                        if score_2 > score_1 and label_2 != 0: 
                            top_relation_list.append((j, i, label_2))

            _, top_roles = role_type_scores[graph_idx].max(1)
            top_roles = top_roles.tolist()
            top_roles = [(i, j, top_roles[i * max_entity_num + j])
                         for i in range(trigger_num) for j in range(entity_num)
                         if top_roles[i * max_entity_num + j] != 0]
            top_graphs.append(Graph(
                entities=top_entities,
                triggers=top_triggers,
                # relations=top_relations,
                relations=top_relation_list,
                roles=top_roles,
                vocabs=graph.vocabs
            ))
        return top_graphs

    def trim_beam_set(self, beam_set, beam_size):
        if len(beam_set) > beam_size:
            beam_set.sort(key=lambda x: self.compute_graph_score(x), reverse=True)
            beam_set = beam_set[:beam_size]
        return beam_set

    def compute_graph_score(self, graph):
        score = graph.graph_local_score
        if self.use_global_features:
            global_vector = generate_global_feature_vector(graph,
                                                           self.global_feature_maps,
                                                           features=self.global_features)
            global_vector = self.global_feature_weights.new_tensor(global_vector)
            global_score = global_vector.dot(self.global_feature_weights).item()
            score = score + global_score
        return score

    def decode(self,
               spans,
               entity_type_scores,
               mention_type_scores,
               event_type_scores,
               relation_type_scores,
               role_type_scores,
               entity_num,
               beam_size=None):
        if type(beam_size) == type(None):
            beam_size = self.beam_size 
        beam_set = [Graph.empty_graph(self.vocabs)]
        entity_idx, trigger_idx = 0, 0

        for start, end, _, is_entity_node in spans:
            # 1. node step
            if is_entity_node:
                node_scores = entity_type_scores[entity_idx].tolist()
            else:
                node_scores = event_type_scores[trigger_idx].tolist()
            node_scores_norm = normalize_score(node_scores)
            node_scores = [(s, i, n) for i, (s, n) in enumerate(zip(node_scores,
                                                                node_scores_norm))]
            node_scores.sort(key=lambda x: x[0], reverse=True)
            top_node_scores = node_scores[:self.beta_v]

            beam_set_ = []
            for graph in beam_set:
                for score, label, score_norm in top_node_scores:
                    graph_ = graph.copy()
                    if is_entity_node:
                        graph_.add_entity(start, end, label, score, score_norm)
                    else:
                        graph_.add_trigger(start, end, label, score, score_norm)
                    beam_set_.append(graph_)
            beam_set = beam_set_

            # 2. edge step
            if is_entity_node:
                # add a new entity: new relations, new argument roles
                for i in range(entity_idx):
                    # add relation edges
                    edge_scores_1 = relation_type_scores[i * entity_num + entity_idx].tolist()
                    edge_scores_2 = relation_type_scores[entity_idx * entity_num + i].tolist()
                    edge_scores_norm_1 = normalize_score(edge_scores_1)
                    edge_scores_norm_2 = normalize_score(edge_scores_2)

                    if self.relation_directional:
                        edge_scores = [(max(s1, s2), n2 if s1 < s2 else n1, i, s1 < s2)
                                       for i, (s1, s2, n1, n2)
                                       in enumerate(zip(edge_scores_1, edge_scores_2,
                                                        edge_scores_norm_1,
                                                        edge_scores_norm_2))]
                        null_score = edge_scores[0][0]
                        edge_scores.sort(key=lambda x: x[0], reverse=True)
                        top_edge_scores = edge_scores[:self.beta_e]
                    else:
                        edge_scores = [(max(s1, s2), n2 if s1 < n2 else n1, i, False)
                                       for i, (s1, s2, n1, n2)
                                       in enumerate(zip(edge_scores_1, edge_scores_2,
                                                        edge_scores_norm_1,
                                                        edge_scores_norm_2))]
                        null_score = edge_scores[0][0]
                        edge_scores.sort(key=lambda x: x[0], reverse=True)
                        top_edge_scores = edge_scores[:self.beta_e]

                    beam_set_ = []
                    for graph in beam_set:
                        has_valid_edge = False
                        for score, score_norm, label, inverse in top_edge_scores:
                            rel_cur_ent = label * 100 + graph.entities[-1][-1]
                            rel_pre_ent = label * 100 + graph.entities[i][-1]
                            if label == 0 or (rel_pre_ent in self.valid_relation_entity and
                                              rel_cur_ent in self.valid_relation_entity):
                                graph_ = graph.copy()
                                if self.relation_directional and inverse:
                                    graph_.add_relation(entity_idx, i, label, score, score_norm)
                                else:
                                    graph_.add_relation(i, entity_idx, label, score, score_norm)
                                beam_set_.append(graph_)
                                has_valid_edge = True
                        if not has_valid_edge:
                            graph_ = graph.copy()
                            graph_.add_relation(i, entity_idx, 0, null_score)
                            beam_set_.append(graph_)
                    beam_set = beam_set_
                    if len(beam_set) > 200:
                        beam_set = self.trim_beam_set(beam_set, beam_size)

                for i in range(trigger_idx):
                    # add argument role edges
                    edge_scores = role_type_scores[i * entity_num + entity_idx].tolist()
                    edge_scores_norm = normalize_score(edge_scores)
                    edge_scores = [(s, i, n) for i, (s, n) in enumerate(zip(edge_scores, edge_scores_norm))]
                    null_score = edge_scores[0][0]
                    edge_scores.sort(key=lambda x: x[0], reverse=True)
                    top_edge_scores = edge_scores[:self.beta_e]

                    beam_set_ = []
                    for graph in beam_set:
                        has_valid_edge = False
                        for score, label, score_norm in top_edge_scores:
                            role_entity = label * 100 + graph.entities[-1][-1]
                            event_role = graph.triggers[i][-1] * 100 + label
                            if label == 0 or (event_role in self.valid_event_role and
                                              role_entity in self.valid_role_entity):
                                graph_ = graph.copy()
                                graph_.add_role(i, entity_idx, label, score, score_norm)
                                beam_set_.append(graph_)
                                has_valid_edge = True
                        if not has_valid_edge:
                            graph_ = graph.copy()
                            graph_.add_role(i, entity_idx, 0, null_score)
                            beam_set_.append(graph_)
                    beam_set = beam_set_
                    if len(beam_set) > 100:
                        beam_set = self.trim_beam_set(beam_set, beam_size)
                beam_set = self.trim_beam_set(beam_set_, beam_size)

            else:
                # add a new trigger: new argument roles
                for i in range(entity_idx):
                    edge_scores = role_type_scores[trigger_idx * entity_num + i].tolist()
                    edge_scores_norm = normalize_score(edge_scores)
                    edge_scores = [(s, i, n) for i, (s, n) in enumerate(zip(edge_scores,
                                                                            edge_scores_norm))]
                    null_score = edge_scores[0][0]
                    edge_scores.sort(key=lambda x: x[0], reverse=True)
                    top_edge_scores = edge_scores[:self.beta_e]

                    beam_set_ = []
                    for graph in beam_set:
                        has_valid_edge = False
                        for score, label, score_norm in top_edge_scores:
                            event_role = graph.triggers[-1][-1] * 100 + label
                            role_entity = label * 100 + graph.entities[i][-1]
                            if label == 0 or (event_role in self.valid_event_role
                                              and role_entity in self.valid_role_entity):
                                graph_ = graph.copy()
                                graph_.add_role(trigger_idx, i, label, score, score_norm)
                                beam_set_.append(graph_)
                                has_valid_edge = True
                        if not has_valid_edge:
                            graph_ = graph.copy()
                            graph_.add_role(trigger_idx, i, 0, null_score)
                            beam_set_.append(graph_)
                    beam_set = beam_set_
                    if len(beam_set) > 100:
                        beam_set = self.trim_beam_set(beam_set, self.beam_size)

                beam_set = self.trim_beam_set(beam_set_, self.beam_size)

            if is_entity_node:
                entity_idx += 1
            else:
                trigger_idx += 1
        beam_set.sort(key=lambda x: self.compute_graph_score(x), reverse=True)
        graph = beam_set[0]

        # predict mention types
        _, mention_types = mention_type_scores.max(dim=1)
        mention_types = mention_types[:entity_idx]
        mention_list = [(i, j, l.item()) for (i, j, k), l
                        in zip(graph.entities, mention_types)]
        graph.mentions = mention_list

        return graph

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
        
