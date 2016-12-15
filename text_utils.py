'''
Created on Dec, 2016

@author: hugo

'''

import os
import json
import numpy as np
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer


####################
# English stopwords
###################
def load_stopwords(file):
    stop_words = []
    try:
        with open(file, 'r') as f:
            for line in f:
                stop_words.append(line.strip('\n '))
    except Exception as e:
        raise e

    return stop_words


###############
# Save and load
###############
def save_json(data, file):
    try:
        with open(file, 'w') as datafile:
            json.dump(data, datafile)
    except Exception as e:
        raise e

def load_json(file):
    try:
        with open(file, 'r') as datafile:
            data = json.load(datafile)
    except Exception as e:
        raise e

    return data

def load_pickle(path_to_file):
    try:
        with open(path_to_file, 'r') as f:
            data = pickle.load(f)
    except Exception as e:
        raise e

    return data

def dump_pickle(data, path_to_file):
    try:
        with open(path_to_file, 'w') as f:
            pickle.dump(data, f)
    except Exception as e:
        raise e

########################
# Construct corpus data
########################
def get_all_files(corpus_path, recursive=False):
    '''Get all files in the directory
        Mode: recursive or not.
    '''
    if recursive:
        return [os.path.join(root, file) for root, dirnames, filenames in os.walk(corpus_path) for file in filenames if not file.startswith('.')]
    else:
        return [os.path.join(corpus_path, filename) for filename in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, filename)) and not filename.startswith('.')]

def save_corpus(out_corpus, doc_word_freq, vocab_dict, word_freq):
    docs = {}
    for filename, val in doc_word_freq.iteritems():
        word_count = {}
        for word, freq in val.iteritems():
            try:
                word_count[vocab_dict[word]] = freq
            except: # word is not in vocab, i.e., this word should be filtered out
                pass
        docs[filename] = word_count
    corpus = {'docs': docs, 'vocab': vocab_dict, 'word_freq': word_freq}
    save_json(corpus, out_corpus)


def load_data(corpus_path, recursive=False):
    word_tokenizer = RegexpTokenizer(r'[a-zA-Z]+') # match only alphabet characters
    try:
        cached_stop_words = load_stopwords('patterns/stopwords.txt')
        print 'loaded patterns/stopwords.txt'
    except:
        from nltk.corpus import stopwords
        cached_stop_words = stopwords.words("english")
        print 'loaded nltk.corpus.stopwords'

    word_freq = defaultdict(lambda: 0) # count the number of times a word appears in a corpus
    doc_word_freq = defaultdict(dict) # count the number of times a word appears in a doc
    files = get_all_files(corpus_path, recursive)

    for filename in files:
        try:
            with open(filename, 'r') as fp:
                text = fp.read().lower()
                words = word_tokenizer.tokenize(text)
                words = [word for word in words if word not in cached_stop_words]

                for i in range(len(words)):
                    # doc-word frequency
                    basename = os.path.basename(filename)
                    try:
                        doc_word_freq[basename][words[i]] += 1
                    except:
                        doc_word_freq[basename][words[i]] = 1
                    # word frequency
                    word_freq[words[i]] += 1
        except Exception as e:
            raise e

    return word_freq, doc_word_freq

def construct_corpus(corpus_path, out_corpus, threshold=5, recursive=False):
    word_freq, doc_word_freq = load_data(corpus_path, recursive)
    print 'finished loading'
    vocab_dict = get_vocab_dict(word_freq, threshold=threshold, topn=None)
    new_word_freq = dict([(word, freq) for word, freq in word_freq.items() if word in vocab_dict])
    save_corpus(out_corpus, doc_word_freq, vocab_dict, new_word_freq)

def load_corpus(corpus_path):
    corpus = load_json(corpus_path)

    return corpus


####################################
# Doc representation (bag of words)
####################################
def doc2vec(doc, dim):
    vec = np.zeros(dim)
    for idx, val in doc.items():
        vec[int(idx)] = val

    return vec


############
# Doc labels
############
def get_20news_doc_labels(corpus_path):
    doc_labels = defaultdict(list)
    files = get_all_files(corpus_path, True)
    for filename in files:
        label, name = filename.split('/')[-2:]
        doc_labels[name].append(label)

    return doc_labels

def get_8k_doc_labels(doc_names):
    doc_labels = {}
    for doc in doc_names:
        doc_labels[doc] = doc.split('-')[-1].replace('.txt', '')

    return doc_labels


#################
# Vocab
#################

def get_vocab_dict(word_freq, threshold=5, topn=None):
    idx = 0
    vocab_dict = {}
    if topn:
        word_freq = dict(sorted(word_freq.items(), key=lambda d:d[1], reverse=True)[:topn])
    for word, freq in word_freq.iteritems():
        if freq < threshold:
            continue
        vocab_dict[word] = idx
        idx += 1
    return vocab_dict

def get_low_freq_words(word_freq, threshold=5):
    return [word for word, freq in word_freq.iteritems() if freq < threshold]

def idf(docs, dim):
    vec = np.zeros((dim, 1))
    for each_doc in docs:
        for idx in each_doc.keys():
            vec[int(idx)] += 1
    return np.log10(1. + len(docs) / vec)

def vocab_weights(vocab_dict, word_freq, max_=100., ratio=.75):
    weights = np.zeros((len(vocab_dict), 1))
    for word, idx in vocab_dict.items():
        weights[idx] = word_freq[word]

    weights = np.clip(weights / max_, 0., 1.)

    return np.power(weights, ratio)

def vocab_weights_tfidf(vocab_dict, word_freq, docs, max_=100., ratio=.75):
    dim = len(vocab_dict)
    tf_vec = np.zeros((dim, 1))
    for word, idx in vocab_dict.items():
        tf_vec[idx] = 1. + np.log10(word_freq[word]) # log normalization

    idf_vec = idf(docs, dim)
    tfidf_vec = tf_vec * idf_vec

    tfidf_vec = np.clip(tfidf_vec, 0., 4.)
    return np.power(tfidf_vec, ratio)
