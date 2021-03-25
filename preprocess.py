import glob
import os
import pickle
import itertools
from typing import Optional
from nltk.corpus.reader.conll import ConllCorpusReader

ONTONOTES_PATH = '/mnt/DATA/Darbas/KTU/code/BERT-BiLSTM-CRF/ontonotes'
OUTPUT_PATH = 'ontonotes'

def convert_data(input_path: str, output_path: str):
    with open(input_path, 'rb') as f:
        data, tags = pickle.load(f)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent, tag in zip(data, tags):
            for item in zip(sent, tag):
                f.writelines([item[0], ' ', item[1], '\n'])
            f.write("\n")

def convert_ontonotes():
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    convert_data(os.path.join(ONTONOTES_PATH, 'data_train.pkl'), os.path.join(OUTPUT_PATH, 'onto_train.txt'))
    convert_data(os.path.join(ONTONOTES_PATH, 'data_valid.pkl'), os.path.join(OUTPUT_PATH, 'onto_valid.txt'))
    convert_data(os.path.join(ONTONOTES_PATH, 'data_test.pkl'), os.path.join(OUTPUT_PATH, 'onto_test.txt'))

def process_conll_file(fpath: str, replacements: Optional[dict]= None):
    reader = ConllCorpusReader(os.path.dirname(fpath), os.path.basename(fpath), columntypes=['words', 'pos', 'ignore',  'chunk'])
    sents = reader.iob_sents()
    tokens, iobs = [], []
    for sent in sents:
        if len(sent) == 0: continue
        tokens.append([t[0] for t in sent])
        if replacements is not None:
            iobs.append([replacements.get(t[2]) or t[2] for t in sent])
    return tokens, iobs

def _create_dataset(output, output_file):
    x_train = list(itertools.chain.from_iterable(x[0] for x in output))
    y_train = list(itertools.chain.from_iterable(x[1] for x in output))
    del output
    with open(output_file, 'wb') as f:
        pickle.dump((x_train, y_train), f)

def process_conll_corpus(corpus_path: str, output_path: str='.', replacements: Optional[dict]= None):
    DATA_TRAIN_PATH = os.path.join(corpus_path, 'data', 'train')
    DATA_TEST_PATH = os.path.join(corpus_path, 'data', 'test')
    fpaths = lambda x, y: glob.glob(x + f'/**/*.{y}', recursive=True)
    files_train = fpaths(DATA_TRAIN_PATH, 'txt') + fpaths(DATA_TRAIN_PATH, 'conll')
    files_test = fpaths(DATA_TEST_PATH, 'txt') + fpaths(DATA_TEST_PATH, 'conll')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    _create_dataset(list(map(lambda x: process_conll_file(x, replacements), files_train)), os.path.join(output_path, 'data_train.pkl'))
    _create_dataset(list(map(lambda x: process_conll_file(x, replacements), files_test)), os.path.join(output_path, 'data_test.pkl'))

def fix_bio_encoding(tags):
    for i in range(1, len(tags)):
        if tags[0].startswith('I-'):
            tags[0] = tags[0].replace('I-', 'B-')
        elif tags[i-1] == 'O' and tags[i].startswith('I-'):
            tags[i] = tags[i].replace('I-', 'B-')
    return tags

def convert_sec_corpus(corpus_path: str, output_path: str='sec'):
    def update_tags(file):
        with open(file, 'rb') as f:
            data, tags = pickle.load(f)
            list(map(fix_bio_encoding, tags))
        with open(file, 'wb') as f:
            pickle.dump((data, tags), f)

    replacements = {'I-PER': "I-PERSON", "B-PER": "B-PERSON"}
    process_conll_corpus(corpus_path, output_path, replacements)
    update_tags(os.path.join(output_path, 'data_train.pkl'))
    update_tags(os.path.join(output_path, 'data_test.pkl'))
    convert_data(os.path.join(output_path, 'data_train.pkl'), os.path.join(output_path, 'data_train.txt'))
    convert_data(os.path.join(output_path, 'data_test.pkl'), os.path.join(output_path, 'data_test.txt'))
