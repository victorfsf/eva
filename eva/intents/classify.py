from boltons.cacheutils import LRI

__all__ = [
    'get_intent'
]

cache = LRI(max_size=1)


def get_intent(*sents, **kwargs):
    if 'intent' not in cache:
        from eva.intents.train import IntentClassifier
        model_file = kwargs.pop(
            'model', 'intents.model'
        )
        classifier = IntentClassifier()
        cache['intent'] = classifier.load(model_file)
    return cache['intent'].predict(sents)
