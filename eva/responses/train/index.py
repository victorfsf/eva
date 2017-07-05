from collections import defaultdict
from eva.responses.train.mixins import SerializeMixin
from eva.utils import regex_tokenize
from gensim import corpora
from gensim import models
from gensim import similarities
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

__all__ = [
    'LSIndexer',
]


class LSIndexer(SerializeMixin):

    def __init__(self, *args, **kwargs):
        self.channels = defaultdict(dict)
        super().__init__(*args, **kwargs)

    def fit(self, channel, documents, speller=None, num_topics=250):
        ch = self.channels[channel]
        self.stemmer = SnowballStemmer(language='portuguese')
        self.stemmer.stopwords = stopwords.words('portuguese')
        if speller:
            ch['speller'] = speller
        ch['documents'] = documents
        texts = [
            self.transform(channel, document) for document in documents
        ]
        ch['dictionary'] = corpora.Dictionary(texts)
        ch['corpus'] = [
            ch['dictionary'].doc2bow(text)
            for text in texts
        ]
        ch['tfidf'] = models.TfidfModel(ch['corpus'])
        ch['lsi'] = models.LsiModel(
            ch['tfidf'][ch['corpus']],
            id2word=ch['dictionary'],
            num_topics=num_topics
        )
        ch['index'] = similarities.MatrixSimilarity(
            ch['lsi'][ch['corpus']]
        )

    def correct(self, channel, word):
        ch = self.channels[channel]
        if 'speller' in ch:
            return ch['speller'].correct(word)
        return word

    def transform(self, channel, document):
        return [
            self.stemmer.stem(self.correct(channel, word.strip()))
            for word in regex_tokenize(document.lower())
            if word not in self.stemmer.stopwords
        ]

    def similarities(self, channel, document):
        stem = self.transform(channel, document)
        ch = self.channels[channel]
        lsi = ch['lsi'][ch['dictionary'].doc2bow(stem)]
        return [(ch['documents'][x], y) for x, y in sorted(
            enumerate(ch['index'][lsi]),
            key=lambda item: -item[1]
        )]

    def search(self, channel, document):
        similarities = self.similarities(channel, document)
        if similarities:
            return similarities[0][0]

    def get(self, channel, document, ratio=None, limit=None):
        similarities = self.similarities(document)
        if similarities:
            result = [
                s[0] for s in similarities
                if ratio is None or s[1] > ratio
            ]
            return result[:limit] if limit else result

    def __repr__(self):
        return '%s(channels=%s)' % (
            self.__class__.__name__,
            len(self.channels), len([
                y for x, z in self.channels.items()
                for y in z['documents']
            ])
        )
