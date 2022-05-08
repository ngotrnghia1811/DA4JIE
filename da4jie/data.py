import copy
import itertools
import json
import torch
from torch.utils.data import Dataset
from collections import Counter, namedtuple, defaultdict
# from recordclass import recordclass

import re

import tqdm
import trankit

from .modules import Graph
from .stanford import stanford_parser

trankit_p = trankit.Pipeline('english')

instance_fields = [
    'sent_id', 'tokens', 'pieces', 'piece_idxs', 'token_lens', 'attention_mask',
    'entity_label_idxs', 'trigger_label_idxs',
    'entity_type_idxs', 'event_type_idxs',
    'relation_type_idxs', 'role_type_idxs',
    'mention_type_idxs',
    'graph', 'entity_num', 'trigger_num',
    'dep_heads', 'dep_rels',
]

batch_fields = [
    'sent_ids', 'tokens', 'piece_idxs', 'token_lens', 'attention_masks',
    'entity_label_idxs', 'trigger_label_idxs',
    'entity_type_idxs', 'event_type_idxs', 'mention_type_idxs',
    'relation_type_idxs', 'role_type_idxs',
    'graphs', 'token_nums',
    'dep_heads', 'dep_rels',
]

batch_fields_live_eval = [
    'sent_ids', 'tokens', 'piece_idxs', 'token_lens', 'attention_masks',
    'entity_label_idxs', 'trigger_label_idxs',
    'entity_type_idxs', 'event_type_idxs', 'mention_type_idxs',
    'relation_type_idxs', 'role_type_idxs',
    'graphs', 'token_nums',
    'doc_ids', 'starts', 'ends', 'texts', 'ori_insts',
    'dep_heads', 'dep_rels',
]

Instance = namedtuple('Instance', field_names=instance_fields)
Batch = namedtuple('Batch', field_names=batch_fields)
Batch_live_eval = namedtuple('Batch_live_eval', field_names=batch_fields_live_eval)

# Instance = recordclass('Instance', instance_fields)
# Batch = recordclass('Batch', batch_fields)
# Batch_live_eval = recordclass('Batch_live_eval', batch_fields_live_eval)

def remove_overlap_entities(entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []
    id_map = {}
    for entity in entities:
        start, end = entity['start'], entity['end']
        for i in range(start, end):
            if tokens[i]:
                id_map[entity['id']] = tokens[i]  # tokens[i] returns the id of the overlapping entity
                continue
        entities_.append(entity)
        for i in range(start, end):
            tokens[i] = entity['id']
    return entities_, id_map


def get_entity_labels(entities, token_num):
    """Convert entity mentions in a sentence to an entity label sequence with
    the length of token_num
    CHECKED
    :param entities (list): a list of entity mentions.
    :param token_num (int): the number of tokens.
    :return:a sequence of BIO format labels.
    """
    labels = ['O'] * token_num
    for entity in entities:
        start, end = entity['start'], entity['end']
        entity_type = entity['entity_type']
        if any([labels[i] != 'O' for i in range(start, end)]):
            continue
        labels[start] = 'B-{}'.format(entity_type)
        for i in range(start + 1, end):
            labels[i] = 'I-{}'.format(entity_type)
    return labels


def get_trigger_labels(events, token_num):
    """Convert event mentions in a sentence to a trigger label sequence with the
    length of token_num.
    :param events (list): a list of event mentions.
    :param token_num (int): the number of tokens.
    :return: a sequence of BIO format labels.
    """
    labels = ['O'] * token_num
    for event in events:
        trigger = event['trigger']
        start, end = trigger['start'], trigger['end']
        event_type = event['event_type']
        labels[start] = 'B-{}'.format(event_type)
        for i in range(start + 1, end):
            labels[i] = 'I-{}'.format(event_type)
    return labels


def get_relation_types(entities, relations, id_map, directional=False,
                       symmetric=None):
    """Get relation type labels among all entities in a sentence.
    :param entities (list): a list of entity mentions.
    :param relations (list): a list of relation mentions.
    :param id_map (dict): a dict of entity ID mapping.
    :param symmetric (set): a set of symmetric relation types.
    :return: a matrix of relation type labels.
    """
    entity_num = len(entities)
    labels = [['O'] * entity_num for _ in range(
        entity_num)]  # a matrix of labels: L, L[i][j] tells the relation type between entity i and entity j.
    entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
    for relation in relations:
        entity_1 = entity_2 = -1
        for arg in relation['arguments']:
            entity_id = arg['entity_id']
            entity_id = id_map.get(entity_id, entity_id)
            if arg['role'] == 'Arg-1':
                entity_1 = entity_idxs[entity_id]
            elif arg['role'] == 'Arg-2':
                entity_2 = entity_idxs[entity_id]
        if entity_1 == -1 or entity_2 == -1:  # skip this relation
            continue
        labels[entity_1][entity_2] = relation['relation_type']
        if not directional:  # similar to adjacency matrix where we consider i-> j and j -> i, default: directional=false
            labels[entity_2][entity_1] = relation['relation_type']
        if symmetric and relation['relation_type'] in symmetric:  # always symmetrix, cannot set this directional
            labels[entity_2][entity_1] = relation['relation_type']
    return labels


def get_relation_list(entities, relations, id_map, vocab, directional=False,
                      symmetric=None):
    """Get the relation list (used for Graph objects)
    :param entities (list): a list of entity mentions.
    :param relations (list): a list of relation mentions.
    :param id_map (dict): a dict of entity ID mapping.
    :param vocab (dict): a dict of label to label index mapping.
    """
    entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(entities))]
    relation_list = []
    for relation in relations:
        arg_1 = arg_2 = None
        for arg in relation['arguments']:
            if arg['role'] == 'Arg-1':
                arg_1 = entity_idxs[id_map.get(
                    arg['entity_id'], arg['entity_id'])]  # if this entity is not obmitted
            elif arg['role'] == 'Arg-2':
                arg_2 = entity_idxs[id_map.get(
                    arg['entity_id'], arg['entity_id'])]
        if arg_1 is None or arg_2 is None:
            continue
        relation_type = relation['relation_type']
        # sort arg1, arg2 in the ascending order of their positions in the sentence.
        if (not directional and arg_1 > arg_2) or \
                (directional and symmetric and relation_type in symmetric and arg_1 > arg_2):
            arg_1, arg_2 = arg_2, arg_1
        if visited[arg_1][arg_2] == 0:
            relation_list.append((arg_1, arg_2, vocab[relation_type]))
            visited[arg_1][arg_2] = 1

    relation_list.sort(key=lambda x: (x[0], x[1]))
    return relation_list


def get_role_types(entities, events, id_map):
    labels = [['O'] * len(entities) for _ in
              range(len(events))]  # each event has its own argument sequence over entities, NOT tokens!
    entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
    for event_idx, event in enumerate(events):
        for arg in event['arguments']:
            entity_id = arg['entity_id']
            entity_id = id_map.get(entity_id, entity_id)
            entity_idx = entity_idxs[entity_id]
            # if labels[event_idx][entity_idx] != 'O':
            #     print('Conflict argument role {} {} {}'.format(event['trigger']['text'], arg['text'], arg['role']))
            labels[event_idx][entity_idx] = arg['role']
    return labels


def get_role_list(entities, events, id_map, vocab):
    entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(events))]
    role_list = []
    for i, event in enumerate(events):
        for arg in event['arguments']:
            entity_idx = entity_idxs[id_map.get(
                arg['entity_id'], arg['entity_id'])]
            if visited[i][entity_idx] == 0:
                role_list.append((i, entity_idx, vocab[arg['role']]))
                visited[i][entity_idx] = 1
    role_list.sort(key=lambda x: (x[0], x[1]))
    return role_list


def get_coref_types(entities):
    entity_num = len(entities)
    labels = [['O'] * entity_num for _ in range(entity_num)]
    clusters = defaultdict(list)
    for i, entity in enumerate(entities):
        entity_id = entity['entity_id']
        cluster_id = entity_id[:entity_id.rfind('-')]
        clusters[cluster_id].append(i)
    for _, entities in clusters.items():
        for i, j in itertools.combinations(entities, 2):
            labels[i][j] = 'COREF'
            labels[j][i] = 'COREF'
    return labels


def get_coref_list(entities, vocab):
    clusters = defaultdict(list)
    coref_list = []
    for i, entity in enumerate(entities):
        entity_id = entity['entity_id']
        cluster_id = entity_id[:entity_id.rfind('-')]
        clusters[cluster_id].append(i)
    for _, entities in clusters.items():
        for i, j in itertools.combinations(entities, 2):
            if i < j:
                coref_list.append((i, j, vocab['COREF']))
            else:
                coref_list.append((j, i, vocab['COREF']))
    coref_list.sort(key=lambda x: (x[0], x[1]))
    return coref_list


def merge_coref_relation_lists(coref_list, relation_list, entity_num):
    visited = [[0] * entity_num for _ in range(entity_num)]
    merge_list = []
    for i, j, l in coref_list:
        visited[i][j] = 1
        visited[j][i] = 1
        merge_list.append((i, j, l))
    for i, j, l in relation_list:
        assert visited[i][j] == 0 and visited[j][i] == 0
        merge_list.append((i, j, l))
    merge_list.sort(key=lambda x: (x[0], x[1]))


def merge_coref_relation_types(coref_types, relation_types):
    entity_num = len(coref_types)
    labels = copy.deepcopy(coref_types)
    for i in range(entity_num):
        for j in range(entity_num):
            label = relation_types[i][j]
            if label != 0:
                assert labels[i][j] == 0
                labels[i][j] = label
    return labels


class IEDataset(Dataset):
    def __init__(self, args, path, max_length=128, gpu=False, ignore_title=False,
                 relation_mask_self=True, relation_directional=False,
                 coref=False, symmetric_relations=None):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        :param ignore_title (bool): Ignore sentences that are titles (default=False).
        """
        self.args = args
        self.path = path
        self.data = []
        self.gpu = gpu
        self.max_length = max_length
        self.ignore_title = ignore_title  # false
        self.relation_mask_self = relation_mask_self  # true
        self.relation_directional = relation_directional  # false
        self.coref = coref  # false
        if symmetric_relations is None:
            self.symmetric_relations = set()
        else:
            self.symmetric_relations = symmetric_relations  # ["PER-SOC"]

        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def entity_type_set(self):
        type_set = set()
        for inst in self.data:
            for entity in inst['entity_mentions']:
                type_set.add(entity['entity_type'])
        return type_set

    @property
    def event_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                type_set.add(event['event_type'])
        return type_set

    @property
    def relation_type_set(self):
        type_set = set()
        for inst in self.data:
            for relation in inst['relation_mentions']:
                type_set.add(relation['relation_type'])
        return type_set

    @property
    def role_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                for arg in event['arguments']:
                    type_set.add(arg['role'])
        return type_set

    def load_data(self):
        """Load data from file."""
        overlength_num = title_num = 0
        with open(self.path, 'r', encoding='utf-8') as r:
            for line in r:
                inst = json.loads(line)
                inst_len = len(inst['pieces'])
                is_title = inst['sent_id'].endswith('-3') \
                           and inst['tokens'][-1] != '.' \
                           and len(inst['entity_mentions']) == 0
                if self.ignore_title and is_title:
                    title_num += 1
                    continue
                if self.max_length != -1 and inst_len > self.max_length - 2:
                    overlength_num += 1
                    continue
                self.data.append(inst)
                if self.args.debug_mode and len(self.data) > 1000:
                    print('Debug mode: only take 100 datapoints for running...')
                    break

        if overlength_num:
            print('Discarded {} overlength instances'.format(overlength_num))
        if title_num:
            print('Discarded {} titles'.format(title_num))
        print('Loaded {} instances from {}'.format(len(self), self.path))

    def numberize(self, tokenizer, vocabs):
        """Numberize word pieces, labels, etcs.
        :param tokenizer: Bert tokenizer.
        :param vocabs (dict): a dict of vocabularies.
        """
        entity_type_stoi = vocabs['entity_type']
        event_type_stoi = vocabs['event_type']
        relation_type_stoi = vocabs['relation_type']
        role_type_stoi = vocabs['role_type']
        mention_type_stoi = vocabs['mention_type']
        entity_label_stoi = vocabs['entity_label']
        trigger_label_stoi = vocabs['trigger_label']
        deprel_type_stoi = vocabs['deprel_type']

        data = []
        for inst in tqdm.tqdm(self.data):
            tokens = inst['tokens']
            pieces = inst['pieces']
            sent_id = inst['sent_id']
            entities = inst['entity_mentions']
            entities, entity_id_map = remove_overlap_entities(
                entities)  # entity_id_map is the map from a removed entity to its overlapping entity
            entities.sort(key=lambda x: x['start'])
            events = inst['event_mentions']
            events.sort(key=lambda x: x['trigger']['start'])
            relations = inst['relation_mentions']
            token_num = len(tokens)
            token_lens = inst['token_lens']

            # Pad word pieces with special tokens
            piece_idxs = tokenizer.encode(pieces,
                                          add_special_tokens=True,
                                          max_length=self.max_length)
            pad_num = self.max_length - len(piece_idxs)
            attn_mask = [1] * len(piece_idxs) + [0] * pad_num
            piece_idxs = piece_idxs + [0] * pad_num

            # Entity
            # - entity_labels and entity_label_idxs are used for identification
            # - entity_types and entity_type_idxs are used for classification
            # - entity_list is used for graph representation
            entity_labels = get_entity_labels(entities, token_num)
            entity_label_idxs = [entity_label_stoi[l] for l in entity_labels]
            entity_types = [e['entity_type'] for e in entities]
            entity_type_idxs = [entity_type_stoi[l] for l in entity_types]
            entity_list = [(e['start'], e['end'], entity_type_stoi[e['entity_type']])
                           for e in entities]
            # entity_num = len(entity_list)
            mention_types = [e['mention_type'] for e in entities]
            mention_type_idxs = [mention_type_stoi[l] for l in mention_types]
            mention_list = [(i, j, l) for (i, j, k), l
                            in zip(entity_list, mention_type_idxs)]

            # Trigger
            # - trigger_labels and trigger_label_idxs are used for identification
            # - event_types and event_type_idxs are used for classification
            # - trigger_list is used for graph representation
            trigger_labels = get_trigger_labels(events, token_num)
            trigger_label_idxs = [trigger_label_stoi[l]
                                  for l in trigger_labels]
            event_types = [e['event_type'] for e in events]
            event_type_idxs = [event_type_stoi[l] for l in event_types]
            trigger_list = [(e['trigger']['start'], e['trigger']['end'],
                             event_type_stoi[e['event_type']])
                            for e in events]

            # Relation
            relation_types = get_relation_types(entities, relations,
                                                entity_id_map,
                                                directional=self.relation_directional,
                                                symmetric=self.symmetric_relations)  # a matrix of [num entities, num entities] size, entry [i][j] tells the relation betwee entity i and entity j
            relation_type_idxs = [[relation_type_stoi[l] for l in ls]
                                  for ls in relation_types]
            if self.relation_mask_self:  # default: true
                for i in range(len(relation_type_idxs)):
                    relation_type_idxs[i][i] = -100  # entity i and itself doesn't have any relation
            relation_list = get_relation_list(entities, relations,
                                              entity_id_map, relation_type_stoi,
                                              directional=self.relation_directional,  # false
                                              symmetric=self.symmetric_relations)  # ['PER-SOC']

            # Argument role
            role_types = get_role_types(entities, events, entity_id_map)
            role_type_idxs = [[role_type_stoi[l] for l in ls]
                              for ls in role_types]
            role_list = get_role_list(entities, events,
                                      entity_id_map, role_type_stoi)

            # Graph
            graph = Graph(
                entities=entity_list,
                triggers=trigger_list,
                relations=relation_list,
                roles=role_list,
                mentions=mention_list,
                vocabs=vocabs,
            )

            # Depedency
            # sent = ' '.join(tokens)
            deps = trankit_p.posdep(tokens, is_sent=True)
            dep_heads = [i['head'] for i in deps['tokens']]
            dep_rels = [deprel_type_stoi[i['deprel']] for i in deps['tokens']]
            # for s in sent['sentences']:
            #     for t in s['tokens']:
            #         dep_heads.append(t['head'])
            #         dep_rel = deprel_type_stoi[t['deprel']]
            #         dep_rels.append(dep_rel)


            instance = Instance(
                sent_id=sent_id,
                tokens=tokens,
                pieces=pieces,
                piece_idxs=piece_idxs,
                token_lens=token_lens,
                attention_mask=attn_mask,
                entity_label_idxs=entity_label_idxs,
                trigger_label_idxs=trigger_label_idxs,
                entity_type_idxs=entity_type_idxs,
                event_type_idxs=event_type_idxs,
                relation_type_idxs=relation_type_idxs,
                mention_type_idxs=mention_type_idxs,
                role_type_idxs=role_type_idxs,
                graph=graph,
                entity_num=len(entities),
                trigger_num=len(events),
                dep_heads = dep_heads,
                dep_rels = dep_rels,
            )
            data.append(instance)
        self.data = data

    def collate_fn(self, batch):
        batch_piece_idxs = []
        batch_tokens = []
        batch_entity_labels, batch_trigger_labels = [], []
        batch_entity_types, batch_event_types = [], []
        batch_relation_types, batch_role_types = [], []
        batch_mention_types = []
        batch_graphs = []
        batch_token_lens = []
        batch_attention_masks = []

        #NEW: Ling graph
        batch_dep_heads = []
        batch_dep_rels = []

        sent_ids = [inst.sent_id for inst in batch]
        token_nums = [len(inst.tokens) for inst in batch]
        max_token_num = max(token_nums)

        max_entity_num = max([inst.entity_num for inst in batch] + [1])
        max_trigger_num = max([inst.trigger_num for inst in batch] + [1])

        for inst in batch:
            token_num = len(inst.tokens)
            batch_piece_idxs.append(inst.piece_idxs)
            batch_attention_masks.append(inst.attention_mask)
            batch_token_lens.append(inst.token_lens)
            batch_graphs.append(inst.graph)
            batch_tokens.append(inst.tokens)
            # for identification
            batch_entity_labels.append(inst.entity_label_idxs +
                                       [0] * (max_token_num - token_num))
            batch_trigger_labels.append(inst.trigger_label_idxs +
                                        [0] * (max_token_num - token_num))
            # for classification
            batch_entity_types.extend(inst.entity_type_idxs +
                                      [-100] * (max_entity_num - inst.entity_num))
            batch_event_types.extend(inst.event_type_idxs +
                                     [-100] * (max_trigger_num - inst.trigger_num))
            batch_mention_types.extend(inst.mention_type_idxs +
                                       [-100] * (max_entity_num - inst.entity_num))
            for l in inst.relation_type_idxs:  # each iteration corresponds to an entity, which indicates the relations of that entity to other entities in the sentence
                batch_relation_types.extend(
                    l + [-100] * (max_entity_num - inst.entity_num))
            batch_relation_types.extend(
                [-100] * max_entity_num * (max_entity_num - inst.entity_num))
            for l in inst.role_type_idxs:  # each iteraction corresponds to a trigger, which indicates the argument roles of entities in the sentence
                batch_role_types.extend(
                    l + [-100] * (max_entity_num - inst.entity_num))
            batch_role_types.extend(
                [-100] * max_entity_num * (max_trigger_num - inst.trigger_num))

            #QUES: some sent have len(dep) > len(tok)
            #ANSW: just use len(tok) for now
            batch_dep_heads.append(inst.dep_heads +
                                       [-100] * (max_token_num - token_num))
            batch_dep_rels.append(inst.dep_rels +
                                       [0] * (max_token_num - token_num))            

        if self.gpu:
            batch_piece_idxs = torch.LongTensor(batch_piece_idxs).cuda()
            batch_attention_masks = torch.FloatTensor(
                batch_attention_masks).cuda()
            # -------------------
            batch_entity_labels = torch.LongTensor(batch_entity_labels).cuda()
            batch_trigger_labels = torch.LongTensor(batch_trigger_labels).cuda()
            batch_entity_types = torch.LongTensor(batch_entity_types).cuda()
            batch_mention_types = torch.LongTensor(batch_mention_types).cuda()
            batch_event_types = torch.LongTensor(batch_event_types).cuda()
            batch_relation_types = torch.LongTensor(batch_relation_types).cuda()
            batch_role_types = torch.LongTensor(batch_role_types).cuda()

            token_nums = torch.LongTensor(token_nums).cuda()

            batch_dep_heads = torch.LongTensor(batch_dep_heads).cuda()
            batch_dep_rels = torch.LongTensor(batch_dep_rels).cuda()

        else:
            batch_piece_idxs = torch.LongTensor(batch_piece_idxs)
            batch_attention_masks = torch.FloatTensor(batch_attention_masks)
            # -------------------
            batch_entity_labels = torch.LongTensor(batch_entity_labels)
            batch_trigger_labels = torch.LongTensor(batch_trigger_labels)
            batch_entity_types = torch.LongTensor(batch_entity_types)
            batch_mention_types = torch.LongTensor(batch_mention_types)
            batch_event_types = torch.LongTensor(batch_event_types)
            batch_relation_types = torch.LongTensor(batch_relation_types)
            batch_role_types = torch.LongTensor(batch_role_types)

            token_nums = torch.LongTensor(token_nums)

            batch_dep_heads = torch.LongTensor(batch_dep_heads)
            batch_dep_rels = torch.LongTensor(batch_dep_rels)

        return Batch(
            sent_ids=sent_ids,
            tokens=[inst.tokens for inst in batch],
            piece_idxs=batch_piece_idxs,
            token_lens=batch_token_lens,
            attention_masks=batch_attention_masks,
            entity_label_idxs=batch_entity_labels,
            trigger_label_idxs=batch_trigger_labels,
            entity_type_idxs=batch_entity_types,
            mention_type_idxs=batch_mention_types,
            event_type_idxs=batch_event_types,
            relation_type_idxs=batch_relation_types,
            role_type_idxs=batch_role_types,
            graphs=batch_graphs,
            token_nums=token_nums,
            dep_heads=batch_dep_heads,
            dep_rels=batch_dep_rels,
        )


# class IEDatasetLive(Dataset):
#     def __init__(self, args, path):
#         """
#         :param path (str): path to the data file.
#         :param max_length (int): max sentence length.
#         :param gpu (bool): use GPU (default=False).
#         :param ignore_title (bool): Ignore sentences that are titles (default=False).
#         """
#         self.args = args
#         self.path = path
#         self.data = []
#         self.gpu = True
#         self.max_length = 256

#         self.load_data()

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, item):
#         return self.data[item]

#     def load_data(self):
#         """Load data from file."""
#         with open(self.path, 'r', encoding='utf-8') as r:
#             inst_list = json.load(r)
#         for inst in inst_list:
#             doc_id = inst['url'] + '<fourie>' + str(inst['current_index'])
#             all_sents = []
#             for text in inst['body']:
#                 sent_list = stanford_parser['english'].sentence_segmentation(text)
#                 all_sents.extend(sent_list)
#             pseudo_document = '\n'.join([sent['text'] for sent in all_sents])
#             inst['doc_text'] = pseudo_document
#             offset = 0
#             for sent_id, sent in enumerate(all_sents):
#                 for token in sent['tokens']:
#                     token['start'] += offset  # document-level span
#                     token['end'] += offset
#                     assert token['text'] == pseudo_document[token['start']: token['end']]
#                 offset = sent['tokens'][-1]['end'] + 1
#                 # add example
#                 example = {
#                     'doc_id': doc_id,
#                     'sent_id': doc_id + '<fourie>{}'.format(sent_id),
#                     'start': sent['tokens'][0]['start'],
#                     'end': sent['tokens'][-1]['end'],
#                     'text': sent['text'],
#                     'tokens': sent['tokens'],
#                     'ori_inst': inst
#                 }
#                 self.data.append(example)
#         print('Loaded {} sentences from {}'.format(len(self), self.path))

#     def numberize(self, tokenizer, vocabs):
#         """Numberize word pieces, labels, etcs.
#         :param tokenizer: Bert tokenizer.
#         :param vocabs (dict): a dict of vocabularies.
#         """
#         entity_type_stoi = vocabs['entity_type']
#         event_type_stoi = vocabs['event_type']
#         relation_type_stoi = vocabs['relation_type']
#         role_type_stoi = vocabs['role_type']
#         mention_type_stoi = vocabs['mention_type']
#         entity_label_stoi = vocabs['entity_label']
#         trigger_label_stoi = vocabs['trigger_label']

#         data = []
#         skip_num = 0
#         for sent in self.data:
#             tokens = [tkn['text'] for tkn in sent['tokens']]
#             pieces = [tokenizer.tokenize(t) for t in tokens]
#             token_lens = [len(x) for x in pieces]
#             if 0 in token_lens:
#                 skip_num += 1
#                 continue
#             pieces = [p for ps in pieces for p in ps]
#             if len(pieces) == 0:
#                 skip_num += 1
#                 continue

#             # Pad word pieces with special tokens
#             piece_idxs = tokenizer.encode(
#                 pieces,
#                 add_special_tokens=True,
#                 max_length=self.max_length
#             )
#             if len(piece_idxs) > self.max_length:
#                 skip_num += 1
#                 continue
#             pad_num = self.max_length - len(piece_idxs)
#             attn_mask = [1] * len(piece_idxs) + [0] * pad_num
#             piece_idxs = piece_idxs + [0] * pad_num
#             if len(piece_idxs) == 0:
#                 skip_num += 1
#                 continue

#             sent_id = sent['sent_id']
#             entities = []
#             entity_id_map = {}
#             events = []
#             relations = []
#             token_num = len(tokens)

#             # Entity
#             # - entity_labels and entity_label_idxs are used for identification
#             # - entity_types and entity_type_idxs are used for classification
#             # - entity_list is used for graph representation
#             entity_labels = ['O'] * token_num
#             entity_label_idxs = [entity_label_stoi[l] for l in entity_labels]
#             entity_types = []
#             entity_type_idxs = []
#             entity_list = []
#             # entity_num = len(entity_list)
#             mention_types = []
#             mention_type_idxs = []
#             mention_list = []

#             # Trigger
#             trigger_labels = ['O'] * token_num
#             trigger_label_idxs = [trigger_label_stoi[l]
#                                   for l in trigger_labels]
#             event_types = []
#             event_type_idxs = []
#             trigger_list = []

#             # Relation
#             relation_types = [[]]
#             relation_type_idxs = [[]]
#             relation_list = []

#             # Argument role
#             role_types = [[]]
#             role_type_idxs = [[]]
#             role_list = []

#             # Graph
#             graph = Graph(
#                 entities=entity_list,
#                 triggers=trigger_list,
#                 relations=relation_list,
#                 roles=role_list,
#                 mentions=mention_list,
#                 vocabs=vocabs,
#             )
#             graph.set_info(
#                 doc_id=sent['doc_id'],
#                 sent_id=sent['sent_id'],
#                 start=sent['start'],
#                 end=sent['end'],
#                 text=sent['text'],
#                 tokens=sent['tokens'],
#                 ori_inst=sent['ori_inst']
#             )
#             instance = Instance(
#                 sent_id=sent_id,
#                 tokens=sent['tokens'],
#                 pieces=pieces,
#                 piece_idxs=piece_idxs,
#                 token_lens=token_lens,
#                 attention_mask=attn_mask,
#                 entity_label_idxs=entity_label_idxs,
#                 trigger_label_idxs=trigger_label_idxs,
#                 entity_type_idxs=entity_type_idxs,
#                 event_type_idxs=event_type_idxs,
#                 relation_type_idxs=relation_type_idxs,
#                 mention_type_idxs=mention_type_idxs,
#                 role_type_idxs=role_type_idxs,
#                 graph=graph,
#                 entity_num=len(entities),
#                 trigger_num=len(events),
#             )
#             data.append(instance)
#         self.data = data

#     def collate_fn(self, batch):
#         batch_piece_idxs = []
#         batch_tokens = []
#         batch_entity_labels, batch_trigger_labels = [], []
#         batch_entity_types, batch_event_types = [], []
#         batch_relation_types, batch_role_types = [], []
#         batch_mention_types = []
#         batch_graphs = []
#         batch_token_lens = []
#         batch_attention_masks = []

#         sent_ids = [inst.sent_id for inst in batch]
#         token_nums = [len(inst.tokens) for inst in batch]
#         max_token_num = max(token_nums)

#         max_entity_num = max([inst.entity_num for inst in batch] + [1])
#         max_trigger_num = max([inst.trigger_num for inst in batch] + [1])

#         batch_doc_ids, batch_starts, batch_ends, batch_texts = [], [], [], []
#         batch_ori_insts = []
#         for inst in batch:
#             token_num = len(inst.tokens)
#             batch_piece_idxs.append(inst.piece_idxs)
#             batch_attention_masks.append(inst.attention_mask)
#             batch_token_lens.append(inst.token_lens)
#             batch_graphs.append(inst.graph)
#             batch_tokens.append(inst.tokens)
#             # for identification
#             batch_entity_labels.append(inst.entity_label_idxs +
#                                        [0] * (max_token_num - token_num))
#             batch_trigger_labels.append(inst.trigger_label_idxs +
#                                         [0] * (max_token_num - token_num))
#             # for classification
#             batch_entity_types.extend(inst.entity_type_idxs +
#                                       [-100] * (max_entity_num - inst.entity_num))
#             batch_event_types.extend(inst.event_type_idxs +
#                                      [-100] * (max_trigger_num - inst.trigger_num))
#             batch_mention_types.extend(inst.mention_type_idxs +
#                                        [-100] * (max_entity_num - inst.entity_num))
#             for l in inst.relation_type_idxs:  # each iteration corresponds to an entity, which indicates the relations of that entity to other entities in the sentence
#                 batch_relation_types.extend(
#                     l + [-100] * (max_entity_num - inst.entity_num))
#             batch_relation_types.extend(
#                 [-100] * max_entity_num * (max_entity_num - inst.entity_num))
#             for l in inst.role_type_idxs:  # each iteraction corresponds to a trigger, which indicates the argument roles of entities in the sentence
#                 batch_role_types.extend(
#                     l + [-100] * (max_entity_num - inst.entity_num))
#             batch_role_types.extend(
#                 [-100] * max_entity_num * (max_trigger_num - inst.trigger_num))

#             batch_doc_ids.append(inst.graph.doc_id)
#             batch_starts.append(inst.graph.start)
#             batch_ends.append(inst.graph.end)
#             batch_texts.append(inst.graph.text)
#             batch_ori_insts.append(inst.graph.ori_inst)
#         batch_piece_idxs = torch.LongTensor(batch_piece_idxs).cuda()
#         batch_attention_masks = torch.FloatTensor(
#             batch_attention_masks).cuda()
#         # -------------------
#         batch_entity_labels = torch.LongTensor(batch_entity_labels).cuda()
#         batch_trigger_labels = torch.LongTensor(batch_trigger_labels).cuda()
#         batch_entity_types = torch.LongTensor(batch_entity_types).cuda()
#         batch_mention_types = torch.LongTensor(batch_mention_types).cuda()
#         batch_event_types = torch.LongTensor(batch_event_types).cuda()
#         batch_relation_types = torch.LongTensor(batch_relation_types).cuda()
#         batch_role_types = torch.LongTensor(batch_role_types).cuda()

#         token_nums = torch.LongTensor(token_nums).cuda()

#         return Batch_live_eval(
#             sent_ids=sent_ids,
#             tokens=[inst.tokens for inst in batch],
#             piece_idxs=batch_piece_idxs,
#             token_lens=batch_token_lens,
#             attention_masks=batch_attention_masks,
#             entity_label_idxs=batch_entity_labels,
#             trigger_label_idxs=batch_trigger_labels,
#             entity_type_idxs=batch_entity_types,
#             mention_type_idxs=batch_mention_types,
#             event_type_idxs=batch_event_types,
#             relation_type_idxs=batch_relation_types,
#             role_type_idxs=batch_role_types,
#             graphs=batch_graphs,
#             token_nums=token_nums,
#             doc_ids=batch_doc_ids,
#             starts=batch_starts,
#             ends=batch_ends,
#             texts=batch_texts,
#             ori_insts=batch_ori_insts
#         )
