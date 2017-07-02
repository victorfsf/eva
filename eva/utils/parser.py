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


def extract_text(entities, entity_type):
    return [
        x['value'] for x in entities
        if x['type'] == entity_type
    ]
