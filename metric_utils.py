'''
Created on Dec, 2016

@author: hugo

'''
import numpy as np


def calc_accuracy(sys_out, ground):
    assert len(sys_out) == len(ground)
    n = len(sys_out)
    return sum([sys_out[i] == ground[i] for i in range(n)]) / float(n)

def jaccard_sim(a, b):
    a, b = set(a), set(b)
    return 1.*len(a.intersection(b)) / len(a.union(b))

def recall(a, b):
    a, b = set(a), set(b)
    return 1.*len(a.intersection(b)) / len(a)

def apk(actual, predicted, k=10):
    '''
  Computes the average precision at k. Used on MAP calculation
  (mean AP@k over multiple queries).
  '''
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if (p in actual) and (p not in predicted[:i]):
            num_hits += 1.0
            score += num_hits / (i+1.0)

#       if not actual:
#           return 0.0
    return score / min(len(actual), k)


def ndcg2(actual, pred, relevs=None, k=20):
    ''' Normalized Discounted Cummulative Gain. '''

    if not relevs :
        relevs = [2.0]*len(actual)

    pred = pred[:k]
    relevs_dict = {actual[i]: relevs[i] for i in xrange(len(actual))}

    r = [relevs_dict[item] if item in relevs_dict else 0.0 for item in pred]
    print "pred dcg"
    print r
    ideal_r = sorted([relevs_dict[item] for item in actual], reverse=True)[:k]

    idcg = dcg(ideal_r)
    return dcg(r)/idcg if idcg!=0.0 else 0.0


def ndcg(actual, pred, relevs=None, k=20):
    ''' Normalized Discounted Cummulative Gain. '''

    if not relevs :
        relevs = ["R1"]*len(actual)

    pred = pred[:k]
    relevs_values = {"R1":2.0, "R2":1.0}
    relevs_dict = {actual[i]: relevs_values[relevs[i]] for i in xrange(len(actual))}

    r = [relevs_dict[doc_id] if doc_id in relevs_dict else 0.0 for doc_id in pred]
#   ideal_r = sorted(r, reverse=True)
    ideal_r = sorted([relevs_dict[doc_id] for doc_id in actual], reverse=True)[:k]

    idcg = dcg(ideal_r)
    return dcg(r)/idcg if idcg!=0.0 else 0.0


def dcg(relevs):
    ''' Discounted Cummulative Gain. '''

    if len(relevs) == 0 :
        return 0.0

    v = relevs[0]
    for i in xrange(1, len(relevs)) :
        v += relevs[i]/np.log2(i+1)

    return v


def recall_at(actual, pred, k=20) :
    ''' Recall at the top k values. '''
    pred = set(pred[:k])
    actual = set(actual)

    return float(len(actual & pred))/len(actual)


def precision_at(actual, pred, k=20):
    ''' Precision at the top k values. '''
    pred = set(pred[:k])
    actual = set(actual)

    return float(len(actual & pred))/k

def is_hit(true, pred):
    if isinstance(true, str):
        true = [true]
    elif isinstance(true, (list, tuple)):
        pass
    else:
        raise TypeError('Unknown argument true:%s' % true)
    for each_true in true:
        each_true = each_true.split(':')
        for each in each_true:
            if each.lower() in pred:
                return True
    return False

def hit_rank(true, pred):
    if isinstance(true, str):
        true = [true]
    elif isinstance(true, (list, tuple)):
        pass
    else:
        raise TypeError('Unknown argument true:%s' % true)

    true_labels = list(set([y.lower() for x in true for y in x.split(':')]))
    rank = float('inf')
    for each in true_labels:
        if each in pred:
            tmp = pred.index(each)
            if rank > tmp:
                rank = tmp
    return rank if rank != float('inf') else None

def match_at_K(truth, results, K):
    """
    Match@K: The relative number of clusters for which at least one of the top-K labels is correct.

    @params
    truth : dict, key is clus name, value is a list of true labels (we can add labels which we think are correct)
    results : dict, key is clus name, value if a list of predicted labels
    K : the K in the definition.
    """
    hit_count = 0.
    for k, v in results.items():
        if is_hit(truth[k], v[:K]):
            hit_count += 1
    return hit_count / len(results)

def mrr_at_K(truth, results, K):
    """
    Mean Reciprocal Rank (MRR@K): Given an ordered list of K proposed labels for a cluster, the reciprocal rank
    is the inverse of the rank of the first correct label, or zero if no label in the list is correct. The mean
    reciprocal rank at K (MRR@K) is the average of the reciprocal ranks of all clusters.

    @params
    truth : dict, key is clus name, value is a list of true labels (we can add labels which we think are correct)
    results : dict, key is clus name, value if a list of predicted labels
    K : the K in the definition.
    """
    score = 0.
    for k, v in results.items():
        rank = hit_rank(truth[k], v[:K])
        if rank != None:
            score += 1 / (1. + rank)

    return score / len(results)
