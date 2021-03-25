from typing import List, Dict
from flair.datasets.sequence_labeling import ColumnDataset
from flair.data import Span

def extract_entities(file: str):
    columns = {0 : 'text', 1 : 'ner'}
    corpus = ColumnDataset(file, columns)
    results = list()
    for i, sent in enumerate(corpus.sentences):
        spans: List[Span] = sent.get_spans('ner')
        res = dict({'id': i, 'sentence': sent.to_original_text()})
        for span in spans:
            if res.get(span.tag) is None:
                res[span.tag] = list()
            res[span.tag].append(span.text)
        results.append(res)
    return results


def extract_ner(tokens: List[str], tags: List[str]) -> Dict[str, List[str]]:
    outputs = dict()

    def append_entry(type, entity):
        if entity is None or type is None:
            return
        if outputs.get(type) is None:
            outputs[type] = list()
        outputs[type].append(' '.join(entity))

    entries = list(zip(tokens, tags))
    entity, type = [], None
    for i, tok, tag in range(len(entries)-1):
        if entries[i][1] == 'O':
            append_entry(type, entity)
            entity, type = [], None
            continue
        if entries[i][1].startswith(('B-', 'I-')):
            ctype = entries[i][1].split('-')[1]
            if type is None:
                type = ctype
            elif ctype == type:
                entity.append(entries[i][0])
            elif ctype != type:
                # Start of new entity span
                append_entry(type, entity)
                entity, type = [entries[i][0]], ctype
    return outputs

