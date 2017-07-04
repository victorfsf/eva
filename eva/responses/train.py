from eva.config import EVA_PATH
from eva.utils import regex_tokenize
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim import similarities
from gensim import models
from os.path import join
import pickle

__all__ = [
    'LSIndexer'
]


class LSIndexer(object):

    def fit(self, documents, num_topics=250):
        self.documents = documents
        self.stemmer = SnowballStemmer(language='portuguese')
        self.stemmer.stopwords = stopwords.words('portuguese')
        texts = [
            self.transform(document) for document in documents
        ]
        self.dictionary = corpora.Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.tfidf = models.TfidfModel(self.corpus)
        self.lsi = models.LsiModel(
            self.tfidf[self.corpus],
            id2word=self.dictionary,
            num_topics=num_topics
        )
        self.index = similarities.MatrixSimilarity(
            self.lsi[self.corpus]
        )

    def transform(self, document):
        return [
            self.stemmer.stem(word.strip())
            for word in regex_tokenize(document.lower())
            if word not in self.stemmer.stopwords
        ]

    def similarities(self, document):
        stem = self.transform(document)
        lsi = self.lsi[self.dictionary.doc2bow(stem)]
        return [(self.documents[x], y) for x, y in sorted(
            enumerate(self.index[lsi]),
            key=lambda item: -item[1]
        )]

    def search(self, document):
        similarities = self.similarities(document)
        if similarities:
            return similarities[0][0]

    def get(self, document, min_=None, limit=None):
        similarities = self.similarities(document)
        if similarities:
            result = [
                s[0] for s in similarities
                if min_ is None or s[1] > min_
            ]
            return result[:limit] if limit else result

    def save(self, path):
        with open(join(EVA_PATH, path), 'wb') as f:
            pickle.dump(self, f)
        return self

    def load(self, path):
        with open(join(EVA_PATH, path), 'rb') as f:
            instance = pickle.load(f)
            self.__dict__ = instance.__dict__
        return self

    def __repr__(self):
        return '%s(documents=%s)' % (
            self.__class__.__name__,
            len(self.documents)
        )
