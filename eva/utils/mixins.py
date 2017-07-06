from eva.config import EVA_PATH
from os.path import join
import pickle

__all__ = [
    'SerializeMixin',
]


class SerializeMixin(object):

    def save(self, path):
        with open(join(EVA_PATH, 'models', path), 'wb') as f:
            pickle.dump(self, f)
        return self

    def load(self, path):
        with open(join(EVA_PATH, 'models', path), 'rb') as f:
            instance = pickle.load(f)
            self.__dict__ = instance.__dict__
        return self
