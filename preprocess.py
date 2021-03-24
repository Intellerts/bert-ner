import os
import pickle

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
