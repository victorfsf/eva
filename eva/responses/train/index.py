from collections import defaultdict
from difflib import SequenceMatcher
from gensim import corpora
from gensim import models
from gensim import similarities
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from unicodedata import normalize
import pickle
import re


def normalize_ascii(value):
    try:
        return normalize('NFKD', str(value)) \
            .encode('ascii', 'ignore').decode('ascii')
    except Exception:
        return value


def regex_tokenize(sentence):
    return list(filter(None, map(
        normalize_ascii,
        re.split(
            r'\s|\t|\n|\r|\?|\-|\_|\.|\,|\!|\(|\)',
            sentence
        )
    )))


def ratio(a, b):
    return SequenceMatcher(a=a.upper(), b=b.upper()).ratio()


def remove_uf(doc):
    return ' '.join(doc.split()[1:])


class Bunch(dict):

    def __init__(self, **kwargs):
        super(Bunch, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass


class LSIndexer:

    def __init__(self, *args, **kwargs):
        self.sections = defaultdict(Bunch)
        self.spellers = LSSpeller()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return self

    def load(self, path):
        with open(path, 'rb') as f:
            instance = pickle.load(f)
            self.__dict__ = instance.__dict__
        return self

    def fit(self, section, documents, **kwargs):
        sec = self.sections[section]
        self.stemmer = SnowballStemmer(language='portuguese')
        self.stemmer.stopwords = stopwords.words('portuguese')
        sec.documents = documents
        texts = [
            self.transform(section, document)
            for document in documents
        ]
        self.spellers.fit(section, documents)
        self.build(section, texts, **kwargs)

    def build(self, section, texts, **kwargs):
        sec = self.sections[section]
        sec.dictionary = corpora.Dictionary(texts)
        sec.corpus = [
            sec.dictionary.doc2bow(text)
            for text in texts
        ]
        sec.tfidf = models.TfidfModel(sec.corpus)
        sec.lsi = models.LsiModel(
            sec.tfidf[sec.corpus],
            id2word=sec.dictionary,
            num_topics=kwargs.pop('num_topics', 200),
            power_iters=kwargs.pop('power_iters', 2)
        )
        sec.index = similarities.MatrixSimilarity(
            sec.lsi[sec.corpus]
        )

    def correct(self, section, word):
        if section in self.spellers.sections:
            return self.spellers.search(section, word)
        return word

    def transform(self, section, document):
        return [
            self.stemmer.stem(self.correct(section, word.strip()))
            for word in regex_tokenize(document.lower())
            if word not in self.stemmer.stopwords
        ]

    def similarities(self, section, document,
                     weight=0.5, limit=100, min_score=0.1):
        stem = self.transform(section, document)
        sec = self.sections[section]
        lsi = sec.lsi[sec.dictionary.doc2bow(stem)]
        highest_score = sorted((
            (sec.documents[_id], score)
            for _id, score in enumerate(sec.index[lsi])
        ), key=lambda item: -item[1])[:limit]
        return sorted(((
            doc, score, (score + ratio(doc, document) * weight)
        ) for doc, score in highest_score if score >= min_score
        ), key=lambda item: -item[2])

    def search(self, section, document, **kwargs):
        similarities = self.similarities(section, document.strip(), **kwargs)
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


class LSSpeller(LSIndexer):

    def __init__(self, *args, **kwargs):
        self.sections = defaultdict(Bunch)

    def fit(self, section, documents, **kwargs):
        sec = self.sections[section]
        documents = re.findall(
            r'\w+|\d+',
            ' '.join({
                y.strip().lower() for x in documents
                for y in regex_tokenize(x)
                if not y.isdigit()
            })
        )
        sec.documents = documents
        texts = [
            self.transform(section, document)
            for document in documents
        ]
        self.build(section, texts, **kwargs)

    def transform(self, section, document):
        return [word for word in document]
