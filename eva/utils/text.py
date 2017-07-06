import regex as re
import unicodedata


__all__ = ['normalize_ascii', 'regex_tokenize', 'extract_text']


def normalize_ascii(value):
    try:
        return unicodedata.normalize('NFKD', str(value))\
            .encode('ascii', 'ignore').decode('ascii')
    except:
        return value


def regex_tokenize(sentence):
    return list(filter(None, map(
        normalize_ascii,
        re.split(
            r'\s|\t|\n|\r|\?|\-|\_|\.|\,|\!|\(|\)',
            sentence
        )
    )))


def extract_text(entities, entity_type):
    return [
        x['value'] for x in entities
        if x['type'] == entity_type
    ]
