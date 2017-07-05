from boltons.cacheutils import LRI
from eva.responses.train import LSIndexer

cache = LRI(max_size=1)


def __cached_indexer(**kwargs):
    if 'indexer' not in cache:
        indexer = LSIndexer()
        cache.update({
            'indexer': indexer.load(
                kwargs.pop('model', 'index.lsi')
            )
        })
    return cache['indexer']


def search(section, text, **kwargs):
    indexer = __cached_indexer(**kwargs)
    return indexer.search(section, text)


def match(section, text, **kwargs):
    indexer = __cached_indexer(**kwargs)
    return indexer.get(section, text, **kwargs)


def similarities(section, text, **kwargs):
    indexer = __cached_indexer(**kwargs)
    return indexer.similarities(section, text)
