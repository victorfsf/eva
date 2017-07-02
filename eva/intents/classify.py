from boltons.cacheutils import LRI
from eva.config import EVA_PATH
from os.path import join
import pickle

__all__ = [
    'get_intent'
]

cache = LRI(max_size=1)


def get_intent(*sents, **kwargs):
    if 'intent' not in cache:
        model_file = kwargs.pop(
            'model',
            join(EVA_PATH, 'models', 'intents.model')
        )
        with open(model_file, 'rb') as f:
            cache.update({
                'intent': pickle.load(f)
            })
    return cache['intent'].predict(sents)
