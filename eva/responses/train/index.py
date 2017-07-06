from collections import defaultdict
from eva.utils.mixins import SerializeMixin
from eva.responses.train.spell import Speller
from eva.utils import regex_tokenize
from gensim import corpora
from gensim import models
from gensim import similarities
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.datasets.base import Bunch
from toolz import frequencies

__all__ = [
    'LSIndexer',
]


class LSIndexer(SerializeMixin):

    def __init__(self, *args, **kwargs):
        self.sections = defaultdict(Bunch)
        super().__init__(*args, **kwargs)

    def fit(self, section, documents, **kwargs):
        sec = self.sections[section]
        self.stemmer = SnowballStemmer(language='portuguese')
        self.stemmer.stopwords = stopwords.words('portuguese')
        sec.documents = documents
        texts = [
            self.transform(section, document)
            for document in documents
        ]
        sec.speller = Speller(documents)
        frequency = kwargs.pop('frequency', 0)
        texts = self.tokenize(texts, frequency)
        sec.dictionary = corpora.Dictionary(texts)
        sec.corpus = [
            sec.dictionary.doc2bow(text)
            for text in texts
        ]

        sec.tfidf = models.TfidfModel(sec.corpus)
        sec.lsi = models.LsiModel(
            sec.tfidf[sec.corpus],
            id2word=sec.dictionary,
            num_topics=kwargs.pop('num_topics', 250)
        )
        sec.index = similarities.MatrixSimilarity(
            sec.lsi[sec.corpus]
        )

    def tokenize(self, texts, frequency):
        if frequency > 0:
            freq_dict = frequencies([y for x in texts for y in x])
            texts = [x for x in texts if freq_dict[x] > frequency]
        return texts

    def correct(self, section, word):
        sec = self.sections[section]
        if sec.get('speller'):
            return sec.speller.correct(word)
        return word

    def transform(self, section, document):
        return [
            self.stemmer.stem(self.correct(section, word.strip()))
            for word in regex_tokenize(document.lower())
            if word not in self.stemmer.stopwords
        ]

    def similarities(self, section, document):
        stem = self.transform(section, document)
        sec = self.sections[section]
        lsi = sec.lsi[sec.dictionary.doc2bow(stem)]
        return [(sec.documents[x], y) for x, y in sorted(
            enumerate(sec.index[lsi]),
            key=lambda item: -item[1]
        )]

    def search(self, section, document):
        similarities = self.similarities(section, document)
        if similarities:
            return similarities[0][0]
        return None

    def get(self, section, document, ratio=None, limit=None):
        similarities = self.similarities(section, document)
        if similarities:
            result = [
                s[0] for s in similarities
                if ratio is None or s[1] > ratio
            ]
            return result[:limit] if limit else result
        return None

    def __repr__(self):
        return '%s(sections=%s)' % (
            self.__class__.__name__,
            len(self.sections)
        )
