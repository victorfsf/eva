from difflib import SequenceMatcher
from eva.utils import regex_tokenize
from eva.utils.mixins import SerializeMixin
from toolz import frequencies
import regex as re

__all__ = [
    'Speller',
]


class Speller(SerializeMixin):

    def __init__(self, documents, alphabet=None):
        self.words = frequencies(map(
            str.lower,
            re.findall(
                r'\w+|\d+',
                ' '.join({y for x in documents for y in regex_tokenize(x)})
            )
        ))
        self.alphabet = alphabet or (
            'abcdefghijklmnopqrstuvwxyz1234567890'
            'ãâàáäẽêèéëĩîìíïõôòóöûũùúü-'
        )
        super().__init__()

    def get_edits1(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes, transposes = zip(*[
            ((l + r[1:]), (l + r[1] + r[0] + r[2:]) if len(r) > 1 else None)
            for l, r in splits if r
        ])
        replaces = [
            l + c + r[1:] for l, r in splits if r
            for c in self.alphabet
        ]
        inserts = [
            l + c + r for l, r in splits
            for c in self.alphabet
        ]
        return filter(None, set(
            list(deletes + transposes) + replaces + inserts
        ))

    def get_edits2(self, word):
        return (
            e2 for e1 in self.get_edits1(word)
            for e2 in self.get_edits1(e1)
        )

    def get_probability(self, word, guess):
        return self.words.get(word, 0) * SequenceMatcher(
            None, word, guess
        ).ratio()

    def get_known_words(self, words):
        return set(w for w in words if w in self.words)

    def correct(self, word):
        word = word.lower()
        return max(
            self.candidates(word),
            key=lambda x: self.get_probability(word, x)
        )

    def candidates(self, word):
        if word:
            return (
                self.get_known_words([word]) or
                self.get_known_words(self.get_edits1(word)) or
                self.get_known_words(self.get_edits2(word)) or
                [word]
            )
        return [word]
