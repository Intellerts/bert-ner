import argparse
import torch
from flair.datasets.sequence_labeling import ColumnCorpus
from flair.embeddings import (TransformerWordEmbeddings, CharacterEmbeddings, StackedEmbeddings,
                              FlairEmbeddings, WordEmbeddings, RoBERTaEmbeddings, ELMoEmbeddings,
                              XLNetEmbeddings, XLMRobertaEmbeddings)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

BERT_MODEL_DIR = 'FinBERT-FinVocab-Uncased'
CACHE_DIR = 'embeddings'
INPUT_PATH = 'ontonotes'
torch.set_default_tensor_type(torch.FloatTensor)

tagger_config = [
    { 'name': 'finbert-char-ner',
      'embeddings': StackedEmbeddings([
            TransformerWordEmbeddings(BERT_MODEL_DIR, cache_dir=CACHE_DIR),
            CharacterEmbeddings()
        ])
    },
    { 'name': 'finbert-ner',
      'embeddings': TransformerWordEmbeddings(BERT_MODEL_DIR, cache_dir=CACHE_DIR)
    },
    { 'name': 'flair-ner',
      'embeddings': StackedEmbeddings([
          FlairEmbeddings('mix-forward'),
          FlairEmbeddings('mix-backward'),
        ])
    },
    { 'name': 'glove-char-ner',
      'embeddings': StackedEmbeddings([
          TransformerWordEmbeddings(BERT_MODEL_DIR, cache_dir=CACHE_DIR),
          WordEmbeddings('glove')
        ])
    },
    { 'name': 'elmo-ner',
      'embeddings': ELMoEmbeddings()
    },
    { 'name': 'roberta-ner',
      'embeddings': RoBERTaEmbeddings('roberta-large')
    },
    { 'name': 'xlnet-ner',
      'embeddings': XLNetEmbeddings()
    },
    { 'name': 'xlm-roberta-ner',
      'embeddings': XLMRobertaEmbeddings()
    },
]

def train_tagger(output_path, train_file, test_file, dev_file):
    columns = {0 : 'text', 1 : 'ner'}
    corpus = ColumnCorpus(output_path, columns, train_file=train_file, test_file=test_file, dev_file=dev_file)
    corpus._test = [x for x in corpus.test if len(x) > 512]  # Bug in Flair: skip very long sentences to avoid errors!
    tag_dictionary = corpus.make_tag_dictionary(tag_type='ner')
    for config in tagger_config:
        tagger = SequenceTagger(hidden_size=256, embeddings=config['embeddings'], tag_dictionary=tag_dictionary, tag_type='ner', use_crf=True)
        trainer = ModelTrainer(tagger, corpus, use_tensorboard=True)
        trainer.train(config['name'], learning_rate=0.1, mini_batch_size=32, max_epochs=100, embeddings_storage_mode='gpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Input path", default=INPUT_PATH)
    parser.add_argument("--train-file", help="Training file", default='onto_train.txt')
    parser.add_argument("--test-file", help="Testing file", default='onto_test.txt')
    parser.add_argument("--dev-file", help="Development file", default=None)
    args = parser.parse_args()
    train_tagger(args.input_path, args.train_file, args.test_file, args.dev_file)
