from eva.utils import IOBReader
from eva.utils.mixins import SerializeMixin
from functools import partialmethod
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

__all__ = [
    'IntentClassifier'
]


class IntentClassifier(SerializeMixin, LinearSVC):

    def fit(self, **kwargs):
        reader = IOBReader(
            kwargs.pop('path', 'data/iob'),
            test_size=kwargs.pop('test_size', 0.2),
            random_state=kwargs.pop('random_state', 42)
        )
        self.train_features, self.train_labels = zip(*reader.train_set)
        self.test_features, self.test_labels = zip(*reader.test_set)
        # TF-IDF
        self.tfidf = TfidfVectorizer()
        train_tfidf = self.tfidf.fit_transform(
            self.stem_features(self.train_features)  # STEMMING
        )
        return super().fit(train_tfidf, self.train_labels)

    def stem_features(self, features):
        if not hasattr(self, 'stemmer'):
            self.stemmer = SnowballStemmer(language='portuguese')
            self.stemmer.stopwords = set(stopwords.words('portuguese'))
        return [
            ' '.join(
                self.stemmer.stem(x) for x in word_tokenize(f)
                if x not in self.stemmer.stopwords
            ) for f in features
        ]

    def predict(self, features):
        if not hasattr(self, 'tfidf'):
            raise AttributeError(
                'The model must be trained with fit() first.'
            )
        features = self.tfidf.transform(
            self.stem_features(features)
        )
        return super().predict(features)

    def _get_evaluations(self, fn, feature_set=None):
        if feature_set:
            features, labels = zip(*feature_set)
        else:
            features, labels = self.test_features, self.test_labels
        return fn(labels, self.predict(features))

    accuracy = partialmethod(
        _get_evaluations,
        metrics.accuracy_score
    )
    report = partialmethod(
        _get_evaluations,
        metrics.classification_report
    )
    confusion_matrix = partialmethod(
        _get_evaluations,
        metrics.confusion_matrix
    )

    def __repr__(self):
        if not hasattr(self, 'tfidf'):
            return super().__repr__()
        return '%s(accuracy=%s, features=%s, labels=%s)' % (
            self.__class__.__name__,
            self.accuracy(),
            len(self.train_features) + len(self.test_features),
            len(self.classes_)
        )
