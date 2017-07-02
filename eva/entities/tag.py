from boltons.cacheutils import LRI
from eva.config import EVA_PATH
from eva.entities.train import IOBTagger
from nltk import Tree
from nltk.chunk import conlltags2tree
from nltk.tag import CRFTagger
from nltk.tokenize import word_tokenize
from os.path import join
import os

__all__ = [
    'pos_tag', 'iob_tag', 'ne_chunk', 'entity_dict'
]

cache = LRI(max_size=2)


def __cached_tagger(model_file, cache_key, tagger_model):
    if cache_key not in cache:
        tagger = tagger_model()
        tagger.set_model_file(
            os.path.join(
                os.getcwd(), join(EVA_PATH, 'models', model_file)
            )
        )
        cache.update({
            cache_key: tagger
        })
    return cache[cache_key]


def pos_tag(*sents, **kwargs):
    tagger = __cached_tagger(
        kwargs.pop('model', 'pos.model'),
        'pos_tag', CRFTagger
    )
    return [
        tagger.tag(word_tokenize(sent))
        for sent in sents
    ]


def iob_tag(*sents, **kwargs):
    tagger = __cached_tagger(
        kwargs.pop('model', 'iob.model'),
        'iob_tag', IOBTagger
    )
    return [
        [(w, p, i) for (w, p), i in tagger.tag(sent)]
        for sent in pos_tag(*sents)
    ]


def ne_chunk(*sents, **kwargs):
    return [
        conlltags2tree(i) for i in iob_tag(*sents, **kwargs)
    ]


def entity_dict(*sents, **kwargs):
    for tree in ne_chunk(*sents, **kwargs):
        entities = []
        for branch in tree:
            if isinstance(branch, Tree):
                entities.append({
                    'type': branch.label(),
                    'value': ' '.join([
                        x for x, _ in branch.leaves()
                    ])
                })
        yield entities
