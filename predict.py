"""Inference script for DA4JIE models.

Runs a trained model on documents in LTF, TXT, or JSON format and produces
structured information extraction output.

Usage:
    python predict.py -m outputs/best.role.mdl -i data/input/ -o data/output/ --format ltf
"""

import os
import json
import glob
import tqdm
import traceback
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from da4jie.models import OneIE
from da4jie.config import Config
from da4jie.data import IEDatasetEval

cur_dir = os.path.dirname(os.path.realpath(__file__))

format_ext_mapping = {
    'txt': 'txt',
    'ltf': 'ltf.xml',
    'json': 'json',
    'json_single': 'json',
}


def load_model(model_path, device=0, gpu=False, beam_size=5):
    """Load a trained OneIE model from a checkpoint file.

    :param model_path (str): Path to the saved model file.
    :param device (int): GPU device index (default=0).
    :param gpu (bool): Load model onto GPU (default=False).
    :param beam_size (int): Beam search width (default=5).
    :returns: (model, tokenizer, config)
    """
    print('Loading the model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    if isinstance(config, dict):
        config = Config.from_dict(config)
    config.bert_cache_dir = os.path.join(cur_dir, 'bert')
    vocabs = state['vocabs']
    valid_patterns = state['valid']

    model = OneIE(config, vocabs, valid_patterns)
    model.load_state_dict(state['model'])
    model.beam_size = beam_size
    if gpu:
        model.cuda(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model_name,
        cache_dir=config.bert_cache_dir,
        do_lower_case=False,
    )
    return model, tokenizer, config


def predict_document(path, model, tokenizer, config, batch_size=20,
                     max_length=128, gpu=False, input_format='txt',
                     language='english'):
    """Run extraction on a single document.

    :param path (str): Path to the input file.
    :param model: Trained model.
    :param tokenizer: BERT tokenizer.
    :param config: Model config.
    :param batch_size (int): Inference batch size.
    :param max_length (int): Max word-piece length per sentence.
    :param gpu (bool): Use GPU.
    :param input_format (str): Input file format ('txt', 'ltf', 'json').
    :param language (str): Document language.
    :returns: (result list, doc info dict)
    """
    test_set = IEDatasetEval(
        path, max_length=max_length, gpu=gpu,
        input_format=input_format, language=language,
    )
    test_set.numberize(tokenizer)

    info = {
        'doc_id': test_set.doc_id,
        'ori_sent_num': test_set.ori_sent_num,
        'sent_num': len(test_set),
    }

    result = []
    for batch in DataLoader(test_set, batch_size=batch_size, shuffle=False,
                            collate_fn=test_set.collate_fn):
        graphs = model.predict(batch)
        for graph, tokens, sent_id, token_ids in zip(
                graphs, batch.tokens, batch.sent_ids, batch.token_ids):
            graph.clean(
                relation_directional=config.relation_directional,
                symmetric_relations=config.symmetric_relations,
            )
            result.append((sent_id, token_ids, tokens, graph))
    return result, info


def predict(model_path, input_path, output_path, log_path=None, cs_path=None,
            batch_size=50, max_length=128, device=0, gpu=False,
            file_extension='txt', beam_size=5, input_format='txt',
            language='english'):
    """Run information extraction on a directory of documents.

    :param model_path (str): Path to the trained model checkpoint.
    :param input_path (str): Directory containing input documents.
    :param output_path (str): Directory to write JSON output.
    :param log_path (str): Optional path to write a processing log.
    :param cs_path (str): Optional directory for cold-start format output.
    :param batch_size (int): Inference batch size.
    :param max_length (int): Max word-piece length per sentence.
    :param device (int): GPU device index.
    :param gpu (bool): Use GPU.
    :param file_extension (str): Process only files with this extension.
    :param beam_size (int): Beam search width.
    :param input_format (str): Input file format ('txt', 'ltf', 'json').
    :param language (str): Document language.
    """
    if gpu:
        torch.cuda.set_device(device)

    model, tokenizer, config = load_model(
        model_path, device=device, gpu=gpu, beam_size=beam_size)

    os.makedirs(output_path, exist_ok=True)
    file_list = glob.glob(os.path.join(input_path, '*.{}'.format(file_extension)))

    log_writer = open(log_path, 'w', encoding='utf-8') if log_path else None

    progress = tqdm.tqdm(total=len(file_list), ncols=75)
    for f in file_list:
        progress.update(1)
        try:
            doc_result, doc_info = predict_document(
                f, model, tokenizer, config,
                batch_size=batch_size, max_length=max_length,
                gpu=gpu, input_format=input_format, language=language,
            )
            doc_id = doc_info['doc_id']
            out_file = os.path.join(output_path, '{}.json'.format(doc_id))
            with open(out_file, 'w', encoding='utf-8') as w:
                for sent_id, token_ids, tokens, graph in doc_result:
                    output = {
                        'doc_id': doc_id,
                        'sent_id': sent_id,
                        'token_ids': token_ids,
                        'tokens': tokens,
                        'graph': graph.to_dict(),
                    }
                    w.write(json.dumps(output) + '\n')
            if log_writer:
                log_writer.write(json.dumps(doc_info) + '\n')
                log_writer.flush()
        except Exception as e:
            traceback.print_exc()
            if log_writer:
                log_writer.write(json.dumps({'file': f, 'message': str(e)}) + '\n')
                log_writer.flush()
    progress.close()

    if log_writer:
        log_writer.close()

    if cs_path:
        from da4jie.convert import json_to_cs
        print('Converting to cold-start format...')
        json_to_cs(output_path, cs_path)


def main():
    parser = ArgumentParser(description='Run DA4JIE inference.')
    parser.add_argument('-m', '--model_path', required=True,
                        help='Path to the trained model checkpoint.')
    parser.add_argument('-i', '--input_dir', required=True,
                        help='Path to the input directory.')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Path to the output directory (JSON files).')
    parser.add_argument('-l', '--log_path', default=None,
                        help='Path to a processing log file.')
    parser.add_argument('-c', '--cs_dir', default=None,
                        help='Path to the cold-start format output directory.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU.')
    parser.add_argument('-d', '--device', default=0, type=int,
                        help='GPU device index (default=0).')
    parser.add_argument('-b', '--batch_size', default=10, type=int,
                        help='Inference batch size (default=10).')
    parser.add_argument('--max_len', default=128, type=int,
                        help='Max sentence length in word pieces (default=128).')
    parser.add_argument('--beam_size', default=5, type=int,
                        help='Beam size for decoding (default=5).')
    parser.add_argument('--lang', default='english',
                        help='Document language (default=english).')
    parser.add_argument('--format', default='txt',
                        choices=list(format_ext_mapping.keys()),
                        help='Input file format (default=txt).')
    args = parser.parse_args()

    extension = format_ext_mapping.get(args.format, 'ltf.xml')
    predict(
        model_path=args.model_path,
        input_path=args.input_dir,
        output_path=args.output_dir,
        cs_path=args.cs_dir,
        log_path=args.log_path,
        batch_size=args.batch_size,
        max_length=args.max_len,
        device=args.device,
        gpu=args.gpu,
        beam_size=args.beam_size,
        file_extension=extension,
        input_format=args.format,
        language=args.lang,
    )


if __name__ == '__main__':
    main()
