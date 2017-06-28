from eva.core.entities.tag import pos_tag
from glob import glob
from nltk.chunk import conlltags2tree
from os.path import basename
from os.path import join
from os.path import splitext
from sklearn.model_selection import train_test_split
import regex as re


class IOBReader(object):

    def __init__(self, dirname, *args, **kwargs):
        self.dirname = join(dirname, '*.iob')
        self.test_size = kwargs.pop('test_size', 0.2)
        self.random_state = kwargs.pop('random_state', 42)
        self.read()
        super().__init__(*args, **kwargs)

    def read(self):
        self.iob_sents = []
        self.sents = []
        self.feature_set = []
        for filename in glob(self.dirname):
            with open(filename, 'rb') as f:
                tags_re = r'\[(.*?)\]'
                tags_sub = r'\[[A-Z]+\s|\]'
                for sentence in f:
                    sentence = sentence.decode('utf8')
                    text = re.sub(tags_sub, '', sentence).strip('\n').strip()
                    self.sents.append(text)
                    self.feature_set.append(
                        (text, splitext(basename(filename))[0])
                    )
                    tags = []
                    for tag in re.findall(tags_re, sentence):
                        tag, value = tag.split(' ', 1)
                        pos = pos_tag(value)[0]
                        first = [(pos[0][0], pos[0][1], 'B-%s' % tag)]
                        tags.append(
                            first + [(w, t, 'I-%s' % tag) for w, t in pos[1:]]
                        )
                    itags = iter(tags)
                    text = re.sub(tags_re, '[NE]', sentence)
                    text_list = text.split('[NE]')
                    pos = pos_tag(*text_list)
                    iob = []
                    for part in pos if tags else [pos]:
                        tagged_part = [(x, y, 'O') for x, y in part]
                        try:
                            ne = next(itags)
                            iob += tagged_part + ne
                        except StopIteration:
                            iob += tagged_part
                    self.iob_sents.append(iob)
        self.chunked_sents = [conlltags2tree(x) for x in self.iob_sents]
        self.iob_train, self.iob_test = train_test_split(
            [[((w, p), i) for w, p, i in s] for s in self.iob_sents],
            test_size=self.test_size,
            random_state=self.random_state
        )
        self.train_set, self.test_set = train_test_split(
            self.feature_set,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def __repr__(self):
        return '%s(sents=%s)' % (
            self.__class__.__name__, len(self.iob_sents)
        )
