from eva.entities.tag import entity_dict
from eva.intents.classify import get_intent

__all__ = ['parse']


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
