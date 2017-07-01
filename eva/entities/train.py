from nltk.tag.crf import CRFTagger

__all__ = [
    'IOBTagger'
]


class IOBTagger(CRFTagger):

    def _get_features(self, tokens, i):

        def tags_since(sentence, i, *pos_list):
            tags = set()
            for word, pos in sentence[:i]:
                if pos in pos_list:
                    tags = set()
                else:
                    tags.add(pos)
            return '+'.join(sorted(tags))

        def tag_suffixes(length, features):
            if len(word) > length:
                feature_list.append('SUF_' + word[-length:])
                if prevword and len(prevword) > length:
                    feature_list.extend([
                        'PREVSUF_' + prevword[-length:],
                        'PREVSUF+SUF_%s+%s' % (
                            prevword[-length:], word[-length:]
                        )
                    ])
                if nextword and len(nextword) > length:
                    feature_list.extend([
                        'NEXTSUF_' + nextword[-length:],
                        'SUF+NEXTSUF_%s+%s' % (
                            word[-length:], nextword[-length:]
                        )
                    ])
                if prevword and len(prevword) > length and \
                        nextword and len(nextword) > length:
                    feature_list.append('PREVSUF+SUF+NEXTSUF_%s+%s+%s' % (
                        prevword[-length:],
                        word[-length:],
                        nextword[-length:]
                    ))

        token = tokens[i]
        feature_list = []

        if not isinstance(token, tuple):
            return feature_list

        word, pos = token
        prevword = tokens[i - 1][0] if i else None
        nextword = tokens[i + 1][0] if i != len(tokens) - 1 else None
        prevpos = tokens[i - 1][1] if i else '<START>'
        nextpos = tokens[i + 1][1] if i != len(tokens) - 1 else '<END>'

        tag_suffixes(1, feature_list)
        tag_suffixes(2, feature_list)
        tag_suffixes(3, feature_list)

        if word[0].isupper():
            feature_list.append('CAPITALIZATION')

        pos_tags = {
            'ART': 'ARTIGO',
            'PREP+ART': 'ARTIGO',
            'V': 'VERBO',
            'NPROP': 'NOME_PROPRIO',
            'PU': 'PONTUACAO'
        }

        if 'PREP' in pos and pos != 'PREP+ART':
            feature_list.append('PREPOSICAO')
        else:
            for tag, label in pos_tags.items():
                if pos == tag:
                    feature_list.append(label)
                    break

        if word.isdigit():
            feature_list.append('IS_NUMBER')
        elif any(c.isdigit() for c in word):
            feature_list.append('HAS_NUMBER')

        if '/' in word:
            feature_list.append('HAS_DASH')

        tags_since_art = tags_since(tokens, i, 'ART', 'PREP+ART')
        if tags_since_art:
            feature_list.append('TAGS-SINCE-ART_%s' % tags_since_art)

        word_list = [x[0] for x in tokens]
        pos_list = [x[1] for x in tokens]
        if max(set(word_list), key=word_list.count) == word:
            feature_list.append('MOST-USED-WORD')
        if max(set(pos_list), key=pos_list.count) == pos:
            feature_list.append('MOST-USED-POS')

        feature_list.extend([
            'WORD_%s' % word,
            'POS_%s' % pos,
            'WORD+POS_%s+%s' % (word, pos),
            'PREVPOS_%s' % prevpos,
            'NEXTPOS_%s' % nextpos,
            'PREVPOS+POS+NEXTPOS_%s+%s+%s' % (prevpos, pos, nextpos),
            'PREVWORD+WORD+NEXTWORD_%s+%s+%s' % (prevword, word, nextword),
            'POS+NEXTPOS_%s+%s' % (pos, nextpos),
            'WORD+NEXTWORD_%s+%s' % (word, nextword),
            'PREVPOS+POS_%s+%s' % (prevpos, pos),
            'PREVWORD+WORD_%s+%s' % (prevword, word),
        ])

        return feature_list

    def __repr__(self):
        return '%s(model_file=\'%s\')' % (
            self.__class__.__name__,
            self._model_file,
        )
