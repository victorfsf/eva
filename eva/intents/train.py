from eva.utils import IOBReader
from functools import partialmethod
from nltk.stem import SnowballStemmer
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords


class IntentClassifier(LinearSVC):

    def fit(self, path='data/', **kwargs):
        reader = IOBReader(
            path,
            test_size=kwargs.pop('test_size', 0.2),
            random_state=kwargs.pop('random_state', 42)
        )
        self.x_features, self.x_labels = zip(*reader.train_set)
        self.y_features, self.y_labels = zip(*reader.test_set)
        # Stemming
        self.stemmer = SnowballStemmer(language='portuguese')
        self.stemmer.stopwords = set(stopwords.words('portuguese'))
        # TF-IDF
        self.tfidf = TfidfVectorizer()
        x_tfidf = self.tfidf.fit_transform([
            self.stemmer.stem(x) for x in self.x_features
            if x not in self.stemmer.stopwords
        ])
        return super().fit(x_tfidf, self.x_labels)

    def predict(self, features):
        if not hasattr(self, 'tfidf'):
            raise AttributeError(
                'The model must be trained with fit() first.'
            )
        features = self.tfidf.transform([
            self.stemmer.stem(x) for x in features
            if x not in self.stemmer.stopwords
        ])
        return super().predict(features)

    def _get_evaluations(self, fn, feature_set):
        features, labels = zip(*feature_set)
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
            self.accuracy(zip(self.y_features, self.y_labels)),
            len(self.x_features) + len(self.y_features),
            len(set(self.x_labels + self.y_labels))
        )
