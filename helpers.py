from __future__ import print_function

import sys
import numpy as np
from functools import cmp_to_key, partial
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import average_precision_score


def readData(data_path=None, debug_data=False, binary=False, preprocessing="quantile_3", at=10, number_features=136,
             bin_cutoff=1.5, cut_zeros=False):
    """
    Function for reading the letor data
    :param binary: boolean if the labels of the data should be binary
    :param preprocessing: if set QuantileTransformer(output_distribution="normal") is used
    :param at: if the number of documents in the query is less than "at" they will not be 
               taken into account. This is needed for calculating the ndcg@k until k=at.
    :return: list of queries, list of labels and list of query id
    """
    if debug_data:
        path = "test_data.txt"
    elif data_path is not None:
        path = data_path

    x = []
    y = []
    q = []
    for line in open(path):
        s = line.split()
        if binary:
            if int(s[0]) > bin_cutoff:
                y.append(1)
            else:
                y.append(0)
        else:
            y.append(int(s[0]))

        q.append(int(s[1].split(":")[1]))

        x.append(np.zeros(number_features))
        for i in range(number_features):
            x[-1][i] = float(s[i + 2].split(":")[1])

    if preprocessing == "quantile_3":
        x = QuantileTransformer(
            output_distribution="normal").fit_transform(x) / 3
    else:
        x = np.array(x)
    y = np.array([y]).transpose()
    q = np.array(q)
    xt = []
    yt = []

    for qid in np.unique(q):
        cs = []
        if cut_zeros:
            for yy in y[q == qid][:, 0]:
                if yy not in cs:
                    cs.append(yy)
            if len(cs) == 1:
                continue
        xt.append(x[q == qid])
        yt.append(y[q == qid])

    return np.array(xt), np.array(yt), q


def nDCG_cls(estimator, X, y, at=10):
    """
        Function for evaluating the ndcg score
        :param estimator: estimator class
        :param X: array of instances
        :param y: target values of the instances
        :return: ndcg score for the given instances
    """
    prediction = estimator.predict_proba(X)
    sort_idx = np.argsort(np.concatenate(prediction))

    sorted_list = y[sort_idx][::-1]
    yref = sorted(y, reverse=True)

    DCG = 0.
    IDCG = 0.
    for i in range(at):
        DCG += (2 ** sorted_list[i] - 1) / np.log2(i + 2)
        IDCG += (2 ** yref[i] - 1) / np.log2(i + 2)
    nDCG = DCG / IDCG
    return nDCG


def nDCGScorer_cls(estimator, X, y, at=10):
    """
        Function for evaluating the ndcg score over queries
        :param estimator: estimator class
        :param X: array of queries
        :param y: array of target values per query
        :return: ndcg score over the queries
    """
    listOfnDCG = []
    for query, y_query in zip(X, y):
        if len(query) < at:
            k = len(query)
        else:
            k = at
        shuffled = np.random.permutation(y_query.shape[0])
        listOfnDCG.append(nDCG_cls(estimator, query[shuffled], y_query[shuffled], k))
    print("nDCG@" + str(at) + ": " + str(round(float(np.mean(listOfnDCG)), 4)) + " +- " + str(
        round(float(np.std(listOfnDCG)), 4)))
    return float(np.mean(listOfnDCG))


def AvgP_cls(estimator, X, y):
    """
        Function for evaluating the AvgP score
        :param estimator: estimator class
        :param X: array of instances
        :param y: target values of the instances
        :return: AvgP score for the given instances
    """
    return average_precision_score(y, estimator.predict_proba(X))


def MAP_cls(estimator, X, y):
    """
        Function for evaluating the AvgP score over queries
        :param estimator: estimator class
        :param X: array of queries
        :param y: array of target values per query
        :return: AvgP score over the queries
    """
    listOfAvgP = []
    for query, y_query in zip(X, y):
        if sum(y_query) == 0:
            listOfAvgP.append(0.0)
        else:
            listOfAvgP.append(AvgP_cls(estimator, query, y_query))

    print("MAP: " + str(round(float(np.mean(listOfAvgP)), 4)) + " +- " + str(round(float(np.std(listOfAvgP)), 4)))
    return float(np.mean(listOfAvgP))
