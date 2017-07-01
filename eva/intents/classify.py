from boltons.cacheutils import LRI
import pickle

__all__ = [
    'intent'
]

cache = LRI(max_size=1)


def intent(*sents, **kwargs):
    if 'intent' not in cache:
        model_file = kwargs.pop('model', 'models/intents.model')
        with open(model_file, 'rb') as f:
            cache.update({
                'intent': pickle.load(f)
            })
    return cache['intent'].predict(sents)
