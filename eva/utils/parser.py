from eva.entities.tag import entity_dict
from eva.intents.classify import get_intent
from itertools import zip_longest

__all__ = ['parse', 'zip_fill']


def parse(*sents):
    return [
        {
            'entities': entities,
            'intent': intent,
            'raw': sent
        } for entities, intent, sent in zip(
            entity_dict(*sents),
            get_intent(*sents),
            sents
        )
    ]


def zip_fill(*items):
    max_len = len(max(items, key=len))
    for item in items:
        if item and len(item) < max_len:
            for _ in range(0, max_len - len(item)):
                item.append(item[-1])
    return zip_longest(*items)
