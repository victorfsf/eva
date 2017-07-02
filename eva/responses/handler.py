from eva.config import INTENT_MAPPING
from eva.config import EVA_PATH
import importlib
import random
from os.path import join


def handle_response(sentence):
    intent = sentence['intent']
    response_handler = INTENT_MAPPING[intent]
    module = importlib.import_module(response_handler)
    if not hasattr(module, 'run'):
        raise AttributeError(
            'The response module must have a \'run\' function'
        )
    path = join(EVA_PATH, 'responses', intent)
    responses = {}

    with open('%s.ok' % path, 'rb') as f:
        responses['ok'] = [
            x.strip()
            for x in f.read().decode('utf-8').split('\n')
        ]

    with open('%s.error' % path, 'rb') as f:
        responses['error'] = [
            x.strip()
            for x in f.read().decode('utf-8').split('\n')
        ]

    random.seed(sentence['raw'])
    return module.run(
        sentence['entities'],
        [random.choice(responses['ok']),
         random.choice(responses['error'])]
    )
