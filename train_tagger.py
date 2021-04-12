import argparse
import torch
from flair.data import Subset, Dataset
from flair.datasets.sequence_labeling import ColumnCorpus
from flair.embeddings import (TransformerWordEmbeddings, CharacterEmbeddings, StackedEmbeddings,
                              FlairEmbeddings, WordEmbeddings, ELMoEmbeddings)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

BERT_MODEL_DIR = 'FinBERT-FinVocab-Uncased'
CACHE_DIR = 'embeddings'
INPUT_PATH = 'ontonotes'
MAX_LENGTH = 512
torch.set_default_tensor_type(torch.FloatTensor)

tagger_config = [
    { 'name': 'finbert-char-ner',
      'embeddings': StackedEmbeddings([
            TransformerWordEmbeddings(BERT_MODEL_DIR, cache_dir=CACHE_DIR, allow_long_sentences=False),
            CharacterEmbeddings()
        ])
    },
    { 'name': 'finbert-ner',
      'embeddings': TransformerWordEmbeddings(BERT_MODEL_DIR, cache_dir=CACHE_DIR, allow_long_sentences=False)
    },
    { 'name': 'flair-ner',
      'embeddings': StackedEmbeddings([
          FlairEmbeddings('mix-forward'),
          FlairEmbeddings('mix-backward'),
        ])
    },
    { 'name': 'glove-char-ner',
      'embeddings': StackedEmbeddings([
          TransformerWordEmbeddings(BERT_MODEL_DIR, cache_dir=CACHE_DIR, allow_long_sentences=False),
          WordEmbeddings('glove')
        ])
    },
    { 'name': 'elmo-ner',
      'embeddings': ELMoEmbeddings()
    },
    { 'name': 'roberta-ner',
      'embeddings': TransformerWordEmbeddings('roberta-large', cache_dir=CACHE_DIR, allow_long_sentences=False)
    },
    { 'name': 'xlnet-ner',
      'embeddings': TransformerWordEmbeddings("xlnet-large-cased", cache_dir=CACHE_DIR, allow_long_sentences=False)
    },
    { 'name': 'xlm-roberta-ner',
      'embeddings': TransformerWordEmbeddings('xlm-roberta-large', cache_dir=CACHE_DIR, allow_long_sentences=False)
    },
]

def filter_long(dataset):
    # Bug in Flair: skip very long sentences to avoid errors!
    if isinstance(dataset, Subset):
        return [x for x in dataset.dataset if len(x) <= MAX_LENGTH]
    elif isinstance(dataset, Dataset):
        return [x for x in dataset if len(x) <= MAX_LENGTH]
    return None

def train_tagger(input_path, train_file, test_file, dev_file):
    columns = {0 : 'text', 1 : 'ner'}
    corpus = ColumnCorpus(input_path, columns, train_file=train_file, test_file=test_file, dev_file=dev_file)
    corpus._train = filter_long(corpus.train)
    corpus._dev = filter_long(corpus.dev)
    corpus._test = filter_long(corpus.test)
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
