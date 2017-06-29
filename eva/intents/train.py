from eva.utils import IOBReader
from functools import partialmethod
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


class IntentClassifier(LinearSVC):

    def fit(self, path='data/', **kwargs):
        reader = IOBReader(
            path,
            test_size=kwargs.pop('test_size', 0.2),
            random_state=kwargs.pop('random_state', 42)
        )
        self.features, self.labels = zip(*reader.train_set)
        self.tfidf = TfidfVectorizer()
        tfidf_features = self.tfidf.fit_transform(self.features)
        return super().fit(tfidf_features, self.labels)

    def predict(self, features):
        if not hasattr(self, 'tfidf'):
            raise AttributeError(
                'The model must be trained with fit() first.'
            )
        return super().predict(self.tfidf.transform(features))

    def _get_evaluations(self, fn, feature_set):
        features, labels = zip(*feature_set)
        return fn(labels, self.predict(features))

    accuracy = partialmethod(
        _get_evaluations,
        fn=metrics.accuracy_score
    )
    report = partialmethod(
        _get_evaluations,
        fn=metrics.classification_report
    )
    confusion_matrix = partialmethod(
        _get_evaluations,
        fn=metrics.confusion_matrix
    )
