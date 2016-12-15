'''
Created on Dec, 2016

@author: hugo

'''

import gensim


def get_wordemb(emb_file, vocab):
    pt = PreTrainEmbedding(emb_file, None)
    vocab_dict = {}

    i = 0.
    for each in vocab:
        emb = pt.get_embedding(each)
        if not emb is None:
            vocab_dict[each] = emb
            i += 1
    print 'get_wordemb hit ratio: %s' % (i / len(vocab))

    return vocab_dict

def get_wordemb2(emb_file, vocab):
    pt = PreTrainEmbedding(emb_file, None)
    vocab_dict = {}

    for each in vocab:
        core = each.lower()
        emb = pt.get_embedding(core)
        if emb == None:
            try:
                emb = np.average(np.r_[[pt.get_embedding(x) for x in core.replace(' ', '_').split('_') if x in pt.model]], axis=0)
            except Exception as e:
                raise e
        if type(emb) == type(np.zeros(1)):
            vocab_dict[each] = emb

    return vocab_dict

class PreTrainEmbedding():
    def __init__(self, file):
        self.model = gensim.models.Word2Vec.load_word2vec_format(file, binary=True)

    def get_embedding(self, word):
        word_list = [word, word.upper(), word.lower(), string.capwords(word, '_')]

        tokens = word.split('_')
        if len(tokens) > 1:
            word_list.append(tokens[-1].upper())
            word_list.append(tokens[-1].lower())

        for w in word_list:
            try:
                result = self.model[w]
                return result
            except KeyError:
                #print 'Can not get embedding for ', w
                continue
        return None
