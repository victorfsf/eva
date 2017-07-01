from eva.entities.tag import entity_dict
from eva.intents.classify import intent

__all__ = ['parse']


def parse(*sents):
    return [
        {
            'entities': next(entity_dict(sent)),
            'intent': intent(sent)[0]
        } for sent in sents
    ]
