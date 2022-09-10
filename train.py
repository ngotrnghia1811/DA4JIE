"""Training script for DA4JIE models.

Supports training OneIE, FourIE, and AGIE (DA4JIE) models for Joint
Information Extraction with optional domain adaptation.

Usage:
    python train.py -c configs/example_oneie.json
"""

import os
import json
import time
import logging
from argparse import ArgumentParser

import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from da4jie.models import OneIE, FourIE, AGIE
from da4jie.modules import Graph
from da4jie.config import Config
from da4jie.data import IEDataset
from da4jie.scorer import score_graphs
from da4jie.util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task

logger = logging.getLogger(__name__)


MODEL_MAP = {
    'oneie': OneIE,
    'fourie': FourIE,
    'agie': AGIE,
}


def main():
    # ── Configuration ─────────────────────────────────────────────────────────
    parser = ArgumentParser(description='Train a DA4JIE model.')
    parser.add_argument('-c', '--config', default='configs/example_oneie.json',
                        help='Path to JSON configuration file.')
    parser.add_argument('--model', default=None,
                        choices=list(MODEL_MAP.keys()),
                        help='Model type (overrides config if set).')
    args = parser.parse_args()
    config = Config.from_json_file(args.config)

    # ── Device setup ──────────────────────────────────────────────────────────
    use_gpu = config.use_gpu and torch.cuda.is_available()
    if use_gpu and config.gpu_device >= 0:
        torch.cuda.set_device(config.gpu_device)

    # ── Output directory ──────────────────────────────────────────────────────
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    output_dir = os.path.join(config.log_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'log.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(config.to_dict()) + '\n')
    print('Log file: {}'.format(log_file))

    best_role_model = os.path.join(output_dir, 'best.role.mdl')
    dev_result_file = os.path.join(output_dir, 'result.dev.json')
    test_result_file = os.path.join(output_dir, 'result.test.json')

    # ── Data loading ──────────────────────────────────────────────────────────
    model_name = config.bert_model_name
    tokenizer = BertTokenizer.from_pretrained(
        model_name, cache_dir=config.bert_cache_dir, do_lower_case=False)

    train_set = IEDataset(
        config.train_file, gpu=use_gpu,
        relation_mask_self=config.relation_mask_self,
        relation_directional=config.relation_directional,
        symmetric_relations=config.symmetric_relations,
        ignore_title=config.ignore_title,
    )
    dev_set = IEDataset(
        config.dev_file, gpu=use_gpu,
        relation_mask_self=config.relation_mask_self,
        relation_directional=config.relation_directional,
        symmetric_relations=config.symmetric_relations,
    )
    test_set = IEDataset(
        config.test_file, gpu=use_gpu,
        relation_mask_self=config.relation_mask_self,
        relation_directional=config.relation_directional,
        symmetric_relations=config.symmetric_relations,
    )

    vocabs = generate_vocabs([train_set, dev_set, test_set])
    train_set.numberize(tokenizer, vocabs)
    dev_set.numberize(tokenizer, vocabs)
    test_set.numberize(tokenizer, vocabs)

    valid_patterns = load_valid_patterns(config.valid_pattern_path, vocabs)

    batch_num = len(train_set) // config.batch_size
    dev_batch_num = (len(dev_set) + config.eval_batch_size - 1) // config.eval_batch_size
    test_batch_num = (len(test_set) + config.eval_batch_size - 1) // config.eval_batch_size

    # ── Model initialization ──────────────────────────────────────────────────
    model_type = (args.model or getattr(config, 'model_type', 'oneie')).lower()
    ModelClass = MODEL_MAP[model_type]
    print(f'Using model: {ModelClass.__name__}')

    model = ModelClass(config, vocabs, valid_patterns)
    if hasattr(model, 'load_bert'):
        model.load_bert(model_name, cache_dir=config.bert_cache_dir)
    if use_gpu:
        model.cuda(device=config.gpu_device)

    # ── Optimizer ────────────────────────────────────────────────────────────
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
            'lr': config.bert_learning_rate,
            'weight_decay': config.bert_weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if not n.startswith('bert') and 'crf' not in n
                       and 'global_feature' not in n],
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if not n.startswith('bert')
                       and ('crf' in n or 'global_feature' in n)],
            'lr': config.learning_rate,
            'weight_decay': 0,
        },
    ]
    optimizer = AdamW(params=param_groups)
    schedule = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=batch_num * config.warmup_epoch,
        num_training_steps=batch_num * config.max_epoch,
    )

    # ── Model state (for checkpointing) ──────────────────────────────────────
    state = dict(
        model=model.state_dict(),
        config=config.to_dict(),
        vocabs=vocabs,
        valid=valid_patterns,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    global_feature_max_step = int(config.global_warmup * batch_num) + 1
    print('Global feature max step:', global_feature_max_step)

    tasks = ['entity', 'trigger', 'relation', 'role']
    best_dev = {k: 0.0 for k in tasks}

    for epoch in range(config.max_epoch):
        print('Epoch: {}'.format(epoch))

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        progress = tqdm.tqdm(total=batch_num, ncols=75,
                             desc='Train {}'.format(epoch))
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(DataLoader(
                train_set,
                batch_size=config.batch_size // config.accumulate_step,
                shuffle=True, drop_last=True,
                collate_fn=train_set.collate_fn)):

            loss = model(batch)
            loss = loss * (1.0 / config.accumulate_step)
            loss.backward()

            if (batch_idx + 1) % config.accumulate_step == 0:
                progress.update(1)
                global_step += 1
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clipping)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
        progress.close()

        # ── Dev evaluation ───────────────────────────────────────────────────
        model.eval()
        progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                             desc='Dev {}'.format(epoch))
        best_dev_role_model = False
        dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []

        with torch.no_grad():
            for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                                    shuffle=False, collate_fn=dev_set.collate_fn):
                progress.update(1)
                graphs = model.predict(batch)
                if config.ignore_first_header:
                    for inst_idx, sent_id in enumerate(batch.sent_ids):
                        if int(sent_id.split('-')[-1]) < 4:
                            graphs[inst_idx] = Graph.empty_graph(vocabs)
                for graph in graphs:
                    graph.clean(
                        relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations,
                    )
                dev_gold_graphs.extend(batch.graphs)
                dev_pred_graphs.extend(graphs)
                dev_sent_ids.extend(batch.sent_ids)
                dev_tokens.extend(batch.tokens)
        progress.close()

        dev_scores = score_graphs(
            dev_gold_graphs, dev_pred_graphs,
            relation_directional=config.relation_directional,
        )
        for task in tasks:
            if dev_scores[task]['f'] > best_dev[task]:
                best_dev[task] = dev_scores[task]['f']
                if task == 'role':
                    print('Saving best role model')
                    state['model'] = model.state_dict()
                    torch.save(state, best_role_model)
                    best_dev_role_model = True
                    save_result(dev_result_file, dev_gold_graphs, dev_pred_graphs,
                                dev_sent_ids, dev_tokens)

        # ── Test evaluation ──────────────────────────────────────────────────
        progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                             desc='Test {}'.format(epoch))
        test_gold_graphs, test_pred_graphs, test_sent_ids, test_tokens = [], [], [], []

        with torch.no_grad():
            for batch in DataLoader(test_set, batch_size=config.eval_batch_size,
                                    shuffle=False, collate_fn=test_set.collate_fn):
                progress.update(1)
                graphs = model.predict(batch)
                if config.ignore_first_header:
                    for inst_idx, sent_id in enumerate(batch.sent_ids):
                        if int(sent_id.split('-')[-1]) < 4:
                            graphs[inst_idx] = Graph.empty_graph(vocabs)
                for graph in graphs:
                    graph.clean(
                        relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations,
                    )
                test_gold_graphs.extend(batch.graphs)
                test_pred_graphs.extend(graphs)
                test_sent_ids.extend(batch.sent_ids)
                test_tokens.extend(batch.tokens)
        progress.close()

        test_scores = score_graphs(
            test_gold_graphs, test_pred_graphs,
            relation_directional=config.relation_directional,
        )

        if best_dev_role_model:
            save_result(test_result_file, test_gold_graphs, test_pred_graphs,
                        test_sent_ids, test_tokens)

        result = json.dumps({'epoch': epoch, 'dev': dev_scores, 'test': test_scores})
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(result + '\n')
        print('Log file:', log_file)

    best_score_by_task(log_file, 'role')


if __name__ == '__main__':
    main()
