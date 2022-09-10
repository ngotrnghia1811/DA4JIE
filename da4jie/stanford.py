import stanfordnlp, os, json, copy
from collections import defaultdict

STANFORD_RESOURCE_DIR = '../resource/stanford'

if not os.path.exists(os.path.join(STANFORD_RESOURCE_DIR, 'en_ewt_models/en_ewt_parser.pt')):
    stanfordnlp.download('en', resource_dir=STANFORD_RESOURCE_DIR, force=True)

if not os.path.exists(os.path.join(STANFORD_RESOURCE_DIR, 'zh_gsd_models/zh_gsd_parser.pt')):
    stanfordnlp.download('zh', resource_dir=STANFORD_RESOURCE_DIR, force=True)

if not os.path.exists(os.path.join(STANFORD_RESOURCE_DIR, 'es_ancora_models/es_ancora_parser.pt')):
    stanfordnlp.download('es', resource_dir=STANFORD_RESOURCE_DIR, force=True)


class Stanford_Parser:
    def __init__(self, language='en'):
        self.lang = language
        self.pretokenized_model = stanfordnlp.Pipeline(lang=language, models_dir=STANFORD_RESOURCE_DIR,
                                                       tokenize_pretokenized=True)
        self.tokenizer = stanfordnlp.Pipeline(lang=language, models_dir=STANFORD_RESOURCE_DIR, processors='tokenize')

    def sentence_segmentation(self, segment_text):
        doc_text = copy.deepcopy(segment_text)
        doc = self.tokenizer(doc_text)

        sentence_list = []
        offset = 0
        for sentence in doc.sentences:
            tokens = {
                'word': [],
                'span': []
            }

            for tok in sentence.tokens:
                for word in tok.words:
                    # ***** get span location in original text *****
                    doc_text, start_char_idx = self.get_startchar_idx(word.text, doc_text)
                    start_char_idx += offset
                    end_char_idx = start_char_idx + len(word.text) - 1
                    offset = end_char_idx + 1

                    tokens['word'].append(segment_text[start_char_idx: end_char_idx + 1])
                    tokens['span'].append((start_char_idx, end_char_idx))
                    assert tokens['word'][-1] == word.text

            if len(tokens['word']) == 0:
                continue

            min_span = min([x[0] for x in tokens['span']])
            max_span = max([x[1] for x in tokens['span']])
            sentence_list.append({
                'text': segment_text[min_span: max_span + 1],
                'tokens': [{'text': word, 'start': span[0] - min_span, 'end': span[1] + 1 - min_span} for word, span in
                           zip(tokens['word'], tokens['span'])]
            })
        return sentence_list

    def get_features(self, ori_words):
        pretokenized_text = ' '.join([word.replace(' ', '-') for word in ori_words])
        ori_text = copy.deepcopy(pretokenized_text)
        tokens = {
            'word': [],
            'lemma': [],
            'upos': [],
            'xpos': [],
            'morph': [],
            'head': [],
            'dep_rel': [],
            'span': []
        }
        if len(pretokenized_text.strip()) == 0:
            return tokens

        doc = self.pretokenized_model(pretokenized_text)

        assert len(doc.sentences) == 1

        offset = 0
        position = 0
        for sentence in doc.sentences:
            for tok in sentence.tokens:
                for word in tok.words:
                    tokens['lemma'].append(word.lemma)
                    tokens['upos'].append(word.upos)
                    tokens['xpos'].append(word.xpos)
                    tokens['morph'].append(word.feats)
                    tokens['head'].append(int(word.governor))
                    tokens['dep_rel'].append(word.dependency_relation.split(':')[0])

                    # ***** get span location in original text *****
                    pretokenized_text, start_char_idx = self.get_startchar_idx(word.text, pretokenized_text)
                    start_char_idx += offset
                    end_char_idx = start_char_idx + len(word.text) - 1

                    tokens['word'].append(ori_text[start_char_idx: end_char_idx + 1])
                    tokens['span'].append((start_char_idx, end_char_idx))

                    assert tokens['word'][-1] == word.text

                    offset = end_char_idx + 1
                    position += 1
        if not len(tokens['word']) == len(tokens['upos']) == len(ori_words):
            return None
        else:
            return tokens

    def get_startchar_idx(self, word, text):
        # ******* search for first non-space character *******
        start_char_idx = 0
        for k in range(len(text)):
            if len(text[k].strip()) > 0:
                start_char_idx = k
                break
        text = text[start_char_idx + len(word):]
        return text, start_char_idx


stanford_parser = {
    'english': Stanford_Parser('en')
}
