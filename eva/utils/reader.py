from eva.entities.tag import pos_tag
from glob import glob
from nltk import word_tokenize
from nltk.chunk import conlltags2tree
from os.path import basename
from os.path import join
from os.path import splitext
from sklearn.model_selection import train_test_split
import regex as re

__all__ = [
    'IOBReader'
]


class IOBReader(object):

    def __init__(self, *args, **kwargs):
        self.dirname = join(kwargs.pop('dirname', 'data/iob'), '*.iob')
        self.test_size = kwargs.pop('test_size', 0.2)
        self.random_state = kwargs.pop('random_state', 42)
        self.read()
        super().__init__(*args, **kwargs)

    def pop_pos(self, pos_tags, word):
        for i, (w, pos) in enumerate(pos_tags):
            if word == w:
                return pos_tags.pop(i)[1]

    def read(self):
        self.iob_sents = []
        self.sents = []
        self.feature_set = []
        self.train_set = []
        self.test_set = []
        self.iob_train = []
        self.iob_test = []
        for filename in glob(self.dirname):
            file_feature_set = []
            file_iob_sents = []
            with open(filename, 'rb') as f:
                tags_re = r'\[(.*?)\]'
                tags_sub = r'\[[A-Z]+\s|\]'
                for sentence in f:
                    sentence = sentence.decode('utf8')
                    text = re.sub(tags_sub, '', sentence).strip('\n').strip()
                    self.sents.append(text)
                    file_feature_set.append(
                        (text, splitext(basename(filename))[0])
                    )

                    pos_tags = pos_tag(text)[0]
                    tags = []
                    for tag in re.findall(tags_re, sentence):
                        tag, value = tag.split(' ', 1)
                        words = word_tokenize(value)
                        first = [(
                            words[0], self.pop_pos(pos_tags, words[0]),
                            'B-%s' % tag
                        )]
                        tags.append(
                            first + [(
                                w, self.pop_pos(pos_tags, w), 'I-%s' % tag
                            ) for w in words[1:]]
                        )
                    itags = iter(tags)
                    text_list = re.sub(tags_re, '[NE]', sentence).split('[NE]')
                    iob = []
                    for part in text_list:
                        tagged_part = [
                            (w, self.pop_pos(pos_tags, w), 'O')
                            for w in word_tokenize(part)
                        ]
                        try:
                            ne = next(itags)
                            iob += tagged_part + ne
                        except StopIteration:
                            iob += tagged_part
                    file_iob_sents.append(iob)
            self.feature_set.extend(file_feature_set)
            self.iob_sents.extend(file_iob_sents)
            file_iob_train, file_iob_test = train_test_split(
                [[((w, p), i) for w, p, i in s] for s in file_iob_sents],
                test_size=self.test_size,
                random_state=self.random_state
            )
            file_train_set, file_test_set = train_test_split(
                file_feature_set,
                test_size=self.test_size,
                random_state=self.random_state
            )
            self.iob_train.extend(file_iob_train)
            self.iob_test.extend(file_iob_test)
            self.train_set.extend(file_train_set)
            self.test_set.extend(file_test_set)

        self.chunked_sents = [
            conlltags2tree(x)
            for x in self.iob_sents
        ]

    def __repr__(self):
        return '%s(sents=%s)' % (
            self.__class__.__name__, len(self.iob_sents)
        )
