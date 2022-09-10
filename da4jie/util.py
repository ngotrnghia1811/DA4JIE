import os
import json
import glob, torch
import lxml.etree as et
from nltk import word_tokenize, sent_tokenize
from copy import deepcopy
import numpy as np
from collections import defaultdict

from .vocabs import data_vocabs

def write_predictions(vocabs, pred_graphs, output_file):
    entity_type_itos = {i: s for s, i in vocabs['entity_type'].items()}
    event_type_itos = {i: s for s, i in vocabs['event_type'].items()}
    relation_type_itos = {i: s for s, i in vocabs['relation_type'].items()}
    role_type_itos = {i: s for s, i in vocabs['role_type'].items()}

    output = defaultdict(list)
    for graph in pred_graphs:
        doc = output[graph.doc_id]
        sentence = {
            'sent_id': graph.sent_id,
            'start': graph.start,
            'end': graph.end,
            'text': graph.text,
            'tokens': graph.tokens,
            'trigger_labels': ['O' for _ in graph.tokens],
            'entity_labels': ['O' for _ in graph.tokens],
            'triggers': [],
            'entities': [],
            'roles': [],
            'relations': [],
            'ori_inst': graph.ori_inst
        }
        doc_text = sentence['ori_inst']['doc_text']
        for trigger in graph.triggers:
            start, end, event_type = trigger
            event_type = event_type_itos[event_type]

            sentence['trigger_labels'][start] = 'B-{}'.format(event_type)
            for i in range(start + 1, end):
                sentence['trigger_labels'][i] = 'I-{}'.format(event_type)
            sentence['triggers'].append({
                'start': start,
                'end': end,
                'text': doc_text[graph.tokens[start]['start']: graph.tokens[end - 1]['end']],
                'event_type': event_type
            })
        for entity in graph.entities:
            start, end, entity_type = entity
            entity_type = entity_type_itos[entity_type]

            sentence['entity_labels'][start] = 'B-{}'.format(entity_type)
            for i in range(start + 1, end):
                sentence['entity_labels'][i] = 'I-{}'.format(entity_type)
            sentence['entities'].append({
                'start': start,
                'end': end,
                'text': doc_text[graph.tokens[start]['start']: graph.tokens[end - 1]['end']],
                'entity_type': entity_type
            })
        for role in graph.roles:
            trigger_index, entity_index, role_type_id = role
            sentence['roles'].append({'trigger_index': trigger_index, 'entity_index': entity_index,
                                      'role_type': role_type_itos[role_type_id]})
        for relation in graph.relations:
            entity1_index, entity2_index, relation_type_id = relation
            sentence['relations'].append({'entity1_index': entity1_index, 'entity2_index': entity2_index,
                                          'relation_type': relation_type_itos[relation_type_id]})

        doc.append(sentence)
    doc_list = []
    for doc_id, sent_list in output.items():
        if len(sent_list) == 0:
            continue
        sent_list.sort(key=lambda x: x['start'])
        url, current_index = doc_id.split('<fourie>')[0], doc_id.split('<fourie>')[1]
        ori_inst = sent_list[0]['ori_inst']
        assert ori_inst['url'] == url and ori_inst['current_index'] == int(current_index)
        doc_annot = {
            'url': url,
            'current_index': int(current_index),
            'doc_text': ori_inst['doc_text'],
            'sentence_list': [{k: v for k, v in sent.items() if k != 'ori_inst'} for sent in sent_list]
        }
        for key in ori_inst:
            if key not in doc_annot:
                doc_annot[key] = ori_inst[key]
        doc_list.append(doc_annot)
    with open(output_file, 'w') as f:
        json.dump(doc_list, f)


def compute_binary_reprs(obj1_reprs, obj2_reprs):  # note that, (obj1, obj2) != (obj2, obj1)
    batch_size, _, rep_dim = obj1_reprs.shape
    num_obj1 = obj1_reprs.shape[1]
    num_obj2 = obj2_reprs.shape[1]
    # cloned_obj1s, cloned_obj2s = [], []
    # for id1 in range(num_obj1):
    #     for id2 in range(num_obj2):
    #         cloned_obj1s.append(obj1_reprs[:, id1])
    #         cloned_obj2s.append(obj2_reprs[:, id2])
    # cloned_obj1s = torch.stack(cloned_obj1s, dim=0).permute(1, 0, 2)  # [batch size, num pairs, rep dim]
    # cloned_obj2s = torch.stack(cloned_obj2s, dim=0).permute(1, 0, 2)  # [batch size, num pairs, rep dim]
    cloned_obj1s = obj1_reprs.repeat(1, 1, num_obj2).view(batch_size, -1, rep_dim)
    cloned_obj2s = obj2_reprs.repeat(1, num_obj1, 1).view(batch_size, -1, rep_dim)
    return cloned_obj1s, cloned_obj2s


def compute_span_reprs(word_reprs, span_idxs):
    '''
    word_reprs.shape: [batch size, num words, word dim]
    span_idxs.shape: [batch size, num spans, 2]
    '''
    batch_span_reprs = []
    batch_size, _, _ = word_reprs.shape
    _, num_spans, _ = span_idxs.shape
    for bid in range(batch_size):
        span_reprs = []
        for sid in range(num_spans):
            start, end = span_idxs[bid][sid]
            words = word_reprs[bid][start: end]  # [span size, word dim]
            span_reprs.append(torch.mean(words, dim=0))
        span_reprs = torch.stack(span_reprs, dim=0)  # [num spans, word dim]
        batch_span_reprs.append(span_reprs)
    batch_span_reprs = torch.stack(batch_span_reprs, dim=0)  # [batch size, num spans, word dim]
    return batch_span_reprs


def compute_word_reprs_first(piece_reprs, first_idxs):
    '''
    piece_reprs.shape: [batch size, max length, rep dim]
    first_idxs.shape: [batch size, num words]
    '''
    batch_word_reprs = []
    batch_size, _, _ = piece_reprs.shape
    _, num_words = first_idxs.shape
    for bid in range(batch_size):
        word_reprs = []
        for wid in range(num_words):
            word_reprs.append(piece_reprs[bid][first_idxs[bid][wid]])
        word_reprs = torch.stack(word_reprs, dim=0)  # [ num words, rep dim]
        batch_word_reprs.append(word_reprs)
    batch_word_reprs = torch.stack(batch_word_reprs, dim=0)  # [batch size, num words, rep dim]
    return batch_word_reprs


def compute_word_reps_avg(piece_reprs, component_idxs):
    '''
        piece_reprs.shape: [batch size, max length, rep dim]
        component_idxs.shape: [batch size, num words, 2]
    '''
    batch_word_reprs = []
    batch_size, _, _ = piece_reprs.shape
    _, num_words, _ = component_idxs.shape
    for bid in range(batch_size):
        word_reprs = []
        for wid in range(num_words):
            wrep = torch.mean(piece_reprs[bid][component_idxs[bid][wid][0]: component_idxs[bid][wid][1]], dim=0)
            word_reprs.append(wrep)
        word_reprs = torch.stack(word_reprs, dim=0)  # [num words, rep dim]
        batch_word_reprs.append(word_reprs)
    batch_word_reprs = torch.stack(batch_word_reprs, dim=0)  # [batch size, num words, rep dim]
    return batch_word_reprs


def generate_vocabs(datasets, coref=False,
                    relation_directional=False,
                    symmetric_relations=None):
    """Generate vocabularies from a list of data sets
    :param datasets (list): A list of data sets
    :return (dict): A dictionary of vocabs
    """
    entity_type_set = set()
    event_type_set = set()
    relation_type_set = set()
    role_type_set = set()
    for dataset in datasets:
        entity_type_set.update(dataset.entity_type_set)
        event_type_set.update(dataset.event_type_set)
        relation_type_set.update(dataset.relation_type_set)
        role_type_set.update(dataset.role_type_set)

    # add inverse relation types for non-symmetric relations
    if relation_directional:
        if symmetric_relations is None:
            symmetric_relations = []
        relation_type_set_ = set()
        for relation_type in relation_type_set:
            relation_type_set_.add(relation_type)
            if relation_directional and relation_type not in symmetric_relations:
                relation_type_set_.add(relation_type + '_inv')

    # entity and trigger labels
    prefix = ['B', 'I']
    entity_label_stoi = {'O': 0}
    trigger_label_stoi = {'O': 0}
    for t in entity_type_set:
        for p in prefix:
            entity_label_stoi['{}-{}'.format(p, t)] = len(entity_label_stoi)
    for t in event_type_set:
        for p in prefix:
            trigger_label_stoi['{}-{}'.format(p, t)] = len(trigger_label_stoi)

    entity_type_stoi = {k: i for i, k in enumerate(entity_type_set, 1)}
    entity_type_stoi['O'] = 0

    event_type_stoi = {k: i for i, k in enumerate(event_type_set, 1)}
    event_type_stoi['O'] = 0

    relation_type_stoi = {k: i for i, k in enumerate(relation_type_set, 1)}
    relation_type_stoi['O'] = 0
    if coref:
        relation_type_stoi['COREF'] = len(relation_type_stoi)

    role_type_stoi = {k: i for i, k in enumerate(role_type_set, 1)}
    role_type_stoi['O'] = 0

    mention_type_stoi = {'NAM': 0, 'NOM': 1, 'PRO': 2, 'UNK': 3}

    return {
        'entity_type': entity_type_stoi,
        'event_type': event_type_stoi,
        'relation_type': relation_type_stoi,
        'role_type': role_type_stoi,
        'mention_type': mention_type_stoi,
        'entity_label': entity_label_stoi,
        'trigger_label': trigger_label_stoi,
    }


def load_valid_patterns(path, vocabs):
    event_type_vocab = vocabs['event_type']
    entity_type_vocab = vocabs['entity_type']
    relation_type_vocab = vocabs['relation_type']
    role_type_vocab = vocabs['role_type']

    # valid event-role
    valid_event_role = set()
    event_role = json.load(
        open(os.path.join(path, 'event_role.json'), 'r', encoding='utf-8'))
    for event, roles in event_role.items():
        if event not in event_type_vocab:
            continue
        event_type_idx = event_type_vocab[event]
        for role in roles:
            if role not in role_type_vocab:
                continue
            role_type_idx = role_type_vocab[role]
            valid_event_role.add(event_type_idx * 100 + role_type_idx)

    # valid relation-entity
    valid_relation_entity = set()
    relation_entity = json.load(
        open(os.path.join(path, 'relation_entity.json'), 'r', encoding='utf-8'))
    for relation, entities in relation_entity.items():
        relation_type_idx = relation_type_vocab[relation]
        for entity in entities:
            entity_type_idx = entity_type_vocab[entity]
            valid_relation_entity.add(
                relation_type_idx * 100 + entity_type_idx)

    # valid role-entity
    valid_role_entity = set()
    role_entity = json.load(
        open(os.path.join(path, 'role_entity.json'), 'r', encoding='utf-8'))
    for role, entities in role_entity.items():
        if role not in role_type_vocab:
            continue
        role_type_idx = role_type_vocab[role]
        for entity in entities:
            entity_type_idx = entity_type_vocab[entity]
            valid_role_entity.add(role_type_idx * 100 + entity_type_idx)

    return {
        'event_role': valid_event_role,
        'relation_entity': valid_relation_entity,
        'role_entity': valid_role_entity
    }


def read_ltf(path):
    root = et.parse(path, et.XMLParser(
        dtd_validation=False, encoding='utf-8')).getroot()
    doc_id = root.find('DOC').get('id')
    doc_tokens = []
    for seg in root.find('DOC').find('TEXT').findall('SEG'):
        seg_id = seg.get('id')
        seg_tokens = []
        seg_start = int(seg.get('start_char'))
        seg_text = seg.find('ORIGINAL_TEXT').text
        for token in seg.findall('TOKEN'):
            token_text = token.text
            start_char = int(token.get('start_char'))
            end_char = int(token.get('end_char'))
            assert seg_text[start_char - seg_start:
                            end_char - seg_start + 1
                   ] == token_text, 'token offset error'
            seg_tokens.append((token_text, start_char, end_char))
        doc_tokens.append((seg_id, seg_tokens))

    return doc_tokens, doc_id


def read_txt(path, language='english'):
    doc_id = os.path.basename(path)
    data = open(path, 'r', encoding='utf-8').read()
    data = [s.strip() for s in data.split('\n') if s.strip()]
    sents = [l for ls in [sent_tokenize(line, language=language) for line in data]
             for l in ls]
    doc_tokens = []
    offset = 0
    for sent_idx, sent in enumerate(sents):
        sent_id = '{}-{}'.format(doc_id, sent_idx)
        tokens = word_tokenize(sent)
        tokens = [(token, offset + i, offset + i + 1)
                  for i, token in enumerate(tokens)]
        offset += len(tokens)
        doc_tokens.append((sent_id, tokens))
    return doc_tokens, doc_id


def read_json(path):
    with open(path, 'r', encoding='utf-8') as r:
        data = [json.loads(line) for line in r]
    doc_id = data[0]['doc_id']
    offset = 0
    doc_tokens = []

    for inst in data:
        tokens = inst['tokens']
        tokens = [(token, offset + i, offset + i + 1)
                  for i, token in enumerate(tokens)]
        offset += len(tokens)
        doc_tokens.append((inst['sent_id'], tokens))
    return doc_tokens, doc_id


def read_json_single(path):
    with open(path, 'r', encoding='utf-8') as r:
        data = [json.loads(line) for line in r]
    doc_id = os.path.basename(path)
    doc_tokens = []
    for inst in data:
        tokens = inst['tokens']
        tokens = [(token, i, i + 1) for i, token in enumerate(tokens)]
        doc_tokens.append((inst['sent_id'], tokens))
    return doc_tokens, doc_id


def save_result(output_file, gold_graphs, pred_graphs, sent_ids, tokens=None):
    with open(output_file, 'w', encoding='utf-8') as w:
        for i, (gold_graph, pred_graph, sent_id) in enumerate(
                zip(gold_graphs, pred_graphs, sent_ids)):
            output = {'sent_id': sent_id,
                      'gold': gold_graph.to_dict(),
                      'pred': pred_graph.to_dict()}
            if tokens:
                output['tokens'] = tokens[i]
            w.write(json.dumps(output) + '\n')


def mention_to_tab(start, end, entity_type, mention_type, mention_id, tokens, token_ids, score=1):
    tokens = tokens[start:end]
    token_ids = token_ids[start:end]
    span = '{}:{}-{}'.format(token_ids[0].split(':')[0],
                             token_ids[0].split(':')[1].split('-')[0],
                             token_ids[1].split(':')[1].split('-')[1])
    mention_text = tokens[0]
    previous_end = int(token_ids[0].split(':')[1].split('-')[1])
    for token, token_id in zip(tokens[1:], token_ids[1:]):
        start, end = token_id.split(':')[1].split('-')
        start, end = int(start), int(end)
        mention_text += ' ' * (start - previous_end) + token
        previous_end = end
    return '\t'.join([
        'json2tab',
        mention_id,
        mention_text,
        span,
        'NIL',
        entity_type,
        mention_type,
        str(score)
    ])


def print_and_log(logger, text, printing=True):
    logger.info(text)
    if printing:
        print(text)


def json_to_mention_results(input_dir, output_dir, file_name,
                            bio_separator=' '):
    mention_type_list = ['nam', 'nom', 'pro', 'nam+nom+pro']
    file_type_list = ['bio', 'tab']
    writers = {}
    for mention_type in mention_type_list:
        for file_type in file_type_list:
            output_file = os.path.join(output_dir, '{}.{}.{}'.format(file_name,
                                                                     mention_type,
                                                                     file_type))
            writers['{}_{}'.format(mention_type, file_type)
            ] = open(output_file, 'w')

    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    for f in json_files:
        with open(f, 'r', encoding='utf-8') as r:
            for line in r:
                result = json.loads(line)
                doc_id = result['doc_id']
                tokens = result['tokens']
                token_ids = result['token_ids']
                bio_tokens = [[t, tid, 'O']
                              for t, tid in zip(tokens, token_ids)]
                # separate bio output
                for mention_type in ['NAM', 'NOM', 'PRO']:
                    tokens_tmp = deepcopy(bio_tokens)
                    for start, end, enttype, mentype in result['graph']['entities']:
                        if mention_type == mentype:
                            tokens_tmp[start] = 'B-{}'.format(enttype)
                            for token_idx in range(start + 1, end):
                                tokens_tmp[token_idx] = 'I-{}'.format(
                                    enttype)
                    writer = writers['{}_bio'.format(mention_type.lower())]
                    for token in tokens_tmp:
                        writer.write(bio_separator.join(token) + '\n')
                    writer.write('\n')
                # combined bio output
                tokens_tmp = deepcopy(bio_tokens)
                for start, end, enttype, _ in result['graph']['entities']:
                    tokens_tmp[start] = 'B-{}'.format(enttype)
                    for token_idx in range(start + 1, end):
                        tokens_tmp[token_idx] = 'I-{}'.format(enttype)
                writer = writers['nam+nom+pro_bio']
                for token in tokens_tmp:
                    writer.write(bio_separator.join(token) + '\n')
                writer.write('\n')
                # separate tab output
                for mention_type in ['NAM', 'NOM', 'PRO']:
                    writer = writers['{}_tab'.format(mention_type.lower())]
                    mention_count = 0
                    for start, end, enttype, mentype in result['graph']['entities']:
                        if mention_type == mentype:
                            mention_id = '{}-{}'.format(doc_id, mention_count)
                            tab_line = mention_to_tab(
                                start, end, enttype, mentype, mention_id, tokens, token_ids)
                            writer.write(tab_line + '\n')
                # combined tab output
                writer = writers['nam+nom+pro_tab']
                mention_count = 0
                for start, end, enttype, mentype in result['graph']['entities']:
                    mention_id = '{}-{}'.format(doc_id, mention_count)
                    tab_line = mention_to_tab(
                        start, end, enttype, mentype, mention_id, tokens, token_ids)
                    writer.write(tab_line + '\n')
    for w in writers:
        w.close()


def normalize_score(scores):
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        return [0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


def print_args(logger, model):
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.parameters():
        n_params = torch.prod(torch.tensor(p.shape)).item()
        if p.requires_grad:
            n_trainable_params += n_params
        else:
            n_nontrainable_params += n_params
    logger.info('-' * 100)
    logger.info('> trainable params: ED model')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info('>>> {0}: {1}'.format(name, param.shape))
    logger.info(
        'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
    logger.info('-' * 100)


def generate_pairwise_idxs(num1, num2):
    """Generate all pairwise combinations among entity mentions (relation) or
    event triggers and entity mentions (argument role).

    For example, if there are 2 triggers and 3 mentions in a sentence, num1 = 2,
    and num2 = 3. We generate the following vector:

    idxs = [0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 1, 4]

    Suppose `trigger_reprs` and `entity_reprs` are trigger/entity representation
    tensors. We concatenate them using:

    te_reprs = torch.cat([entity_reprs, entity_reprs], dim=1)

    After that we select vectors from `te_reprs` using (incomplete code) to obtain
    pairwise combinations of all trigger and entity vectors.

    te_reprs = torch.gather(te_reprs, 1, idxs)
    te_reprs = te_reprs.view(batch_size, -1, 2 * bert_dim)

    :param num1: trigger number (argument role) or entity number (relation)
    :param num2: entity number (relation)
    :return (list): a list of indices
    """
    idxs = []
    for i in range(num1):
        for j in range(num2):
            idxs.append(i)
            idxs.append(j + num1)
    return idxs


def max_pooling3d(mask, reps):
    '''
    take-out positions are marked -1, padding positions are marked 0
    mask.shape = [batch size, seq len]
    reps.shape = [batch size, seq len, rep dim]
    max pooling along axis 1
    '''
    mask = mask.long().eq(0).bool()
    pool = reps.masked_fill_(mask.unsqueeze(2), -np.inf)
    pool = torch.max(pool, dim=1)[0]
    return pool

def best_score_by_task(log_file, task, max_epoch=1000):
    with open(log_file, 'r', encoding='utf-8') as r:
        config = r.readline()

        best_scores = []
        best_dev_score = 0
        for line in r:
            record = json.loads(line)
            dev = record['dev']
            test = record['test']
            epoch = record['epoch']
            if epoch > max_epoch:
                break
            if dev[task]['f'] > best_dev_score:
                best_dev_score = dev[task]['f']
                best_scores = [dev, test, epoch]

        print('Epoch: {}'.format(best_scores[-1]))
        tasks = ['entity', 'mention', 'relation', 'trigger_id', 'trigger',
                 'role_id', 'role']
        for t in tasks:
            print('{}: dev: {:.2f}, test: {:.2f}'.format(t,
                                                         best_scores[0][t][
                                                             'f'] * 100.0,
                                                         best_scores[1][t][
                                                             'f'] * 100.0))
