from eva.responses.train.mixins import SerializeMixin
from toolz import frequencies

__all__ = [
    'Speller',
]


class Speller(SerializeMixin):

    def __init__(self, words, alphabet=None):
        self.words = frequencies(map(str.lower, words))
        self.alphabet = alphabet or (
            'abcdefghijklmnopqrstuvwxyz1234567890'
            'ãâàáäẽêèéëĩîìíïõôòóöûũùúü-'
        )
        super().__init__()

    def get_edits1(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [l + r[1:] for l, r in splits if r]
        transposes = [
            l + r[1] + r[0] + r[2:]
            for l, r in splits if len(r) > 1
        ]
        replaces = [
            l + c + r[1:] for l, r in splits if r
            for c in self.alphabet
        ]
        inserts = [
            l + c + r for l, r in splits
            for c in self.alphabet
        ]
        return set(deletes + transposes + replaces + inserts)

    def get_edits2(self, word):
        return (
            e2 for e1 in self.get_edits1(word)
            for e2 in self.get_edits1(e1)
        )

    def get_probability(self, word, n=None):
        if not n:
            n = sum(self.words.values())
        return self.words[word] / n

    def get_known_words(self, words):
        return set(w for w in words if w in self.words)

    def correct(self, word):
        word = word.lower()
        return max(
            self.candidates(word),
            key=self.get_probability
        )

    def candidates(self, word):
        return (
            self.get_known_words([word]) or
            self.get_known_words(self.get_edits1(word)) or
            self.get_known_words(self.get_edits2(word)) or
            [word]
        )
