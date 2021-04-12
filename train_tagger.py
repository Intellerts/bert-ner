import os
import pickle
import torch
from flair.datasets.sequence_labeling import ColumnCorpus
from flair.embeddings import (TransformerWordEmbeddings, CharacterEmbeddings, StackedEmbeddings,
                              FlairEmbeddings, WordEmbeddings, RoBERTaEmbeddings, ELMoEmbeddings,
                              XLNetEmbeddings, XLMRobertaEmbeddings)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

BERT_MODEL_DIR = 'FinBERT-FinVocab-Uncased'
CACHE_DIR = 'embeddings'
OUTPUT_PATH = 'ontonotes'
torch.set_default_tensor_type(torch.FloatTensor)

columns = {0 : 'text', 1 : 'ner'}
corpus = ColumnCorpus(OUTPUT_PATH, columns, train_file = 'onto_train.txt',
                      test_file = 'onto_test.txt', dev_file = 'onto_valid.txt')
tag_dictionary = corpus.make_tag_dictionary(tag_type='ner')
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
for config in tagger_config:
    tagger = SequenceTagger(hidden_size=256, embeddings=config['embeddings'], tag_dictionary=tag_dictionary, tag_type='ner', use_crf=True)
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(config['name'], learning_rate=0.1, mini_batch_size=32, max_epochs=100, embeddings_storage_mode='gpu')
