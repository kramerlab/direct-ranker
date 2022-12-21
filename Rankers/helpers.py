from __future__ import print_function

import sys
import os
import numpy as np
from functools import cmp_to_key, partial
from sklearn.preprocessing import QuantileTransformer
from sklearn import metrics
from sklearn.metrics import average_precision_score
from Rankers.models.DirectRanker import DirectRanker
from tqdm import tqdm
import random


colors_points = [
    (57 / 255, 106 / 255, 177 / 255),    # blue
    (218 / 255, 124 / 255, 48 / 255),    # orange
    (62 / 255, 150 / 255, 81 / 255),     # green
    (204 / 255, 37 / 255, 41 / 255),     # red
    (83 / 255, 81 / 255, 84 / 255),      # black
    (107 / 255, 76 / 255, 154 / 255),    # purple
    (146 / 255, 36 / 255, 40 / 255),     # wine
    (148 / 255, 139 / 255, 61 / 255)     # gold
]
colors_bar = [
    (114 / 255, 147 / 255, 203 / 255),    # blue
    (225 / 255, 151 / 255, 76 / 255),     # orange
    (132 / 255, 186 / 255, 91 / 255),     # green
    (211 / 255, 94 / 255, 96 / 255),      # red
    (128 / 255, 133 / 255, 133 / 255),    # black
    (144 / 255, 103 / 255, 167 / 255),    # purple
    (171 / 255, 104 / 255, 87 / 255),     # wine
    (204 / 255, 194 / 255, 16 / 255)      # gold
]


def gen_set(n_feats=136, n_relevance=5, n_train=100000, n_test=10000, path="test", sigma_label=0.0, test=False):
    n_feats = n_feats
    if test:
        n_train = 100
        n_test = 100
    else:
        n_train = n_train
        n_test = n_test
    n_relevance = n_relevance
    max_qid_train = 10
    max_qid_test = 20
    means = np.random.rand(n_relevance, n_feats) * 100
    sigmas = np.random.rand(n_relevance, n_feats) * 150 + 50

    if not os.path.exists(path):
        os.makedirs(path)

    f = open(f"{path}/train", "w")
    print("Creating traing set...")
    for i in tqdm(range(n_train // n_relevance)):
        for j in range(n_relevance):
            if sigma_label == 0:
                f.write(str(j))
            else:
                label = int(random.normalvariate(mu=j, sigma=sigma_label))
                if label < 0: label = 0
                if label > 4: label = 4
                f.write(str(label))
            f.write(" qid:" + str(int(random.uniform(0, max_qid_train))))
            for idx, n in enumerate(np.random.randn(n_feats) * sigmas[j] + means[j]):
                f.write(" " + str(idx + 1) + ":" + str(n))
            f.write("\n")
    f.close()

    f = open(f"{path}/test", "w")
    print("Creating test set...")
    for i in tqdm(range(n_test // n_relevance)):
        for j in range(n_relevance):
            if sigma_label == 0:
                f.write(str(j))
            else:
                label = int(random.normalvariate(mu=j, sigma=sigma_label))
                if label < 0: label = 0
                if label > 4: label = 4
                f.write(str(label))
            f.write(" qid:" + str(int(random.uniform(max_qid_train, max_qid_test))))
            for idx, n in enumerate(np.random.randn(n_feats) * sigmas[j] + means[j]):
                f.write(" " + str(idx + 1) + ":" + str(n))
            f.write("\n")

    f.close()

def auc_cls(estimator, X, y, w, cnn_bdt=False, linear=False, reorder=True, use_weights=True):
    if cnn_bdt:
        prediction = estimator.predict_proba(X)[:,1]
    else:
        if linear:
            prediction = estimator.predict(X)
        else:
            prediction = estimator.predict_proba(X)
    if use_weights:
        fpr, tpr, _ = metrics.roc_curve(y, prediction, sample_weight=w, pos_label=1)
    else:
        fpr, tpr, _ = metrics.roc_curve(y, prediction, pos_label=1)
    order = np.lexsort((tpr, fpr))
    fpr, tpr = fpr[order], tpr[order]
    #print("AUC: " + str(metrics.auc(fpr, tpr)))
    return metrics.auc(fpr, tpr)


def auc_value(prediction, y, w, reorder=True):
    fpr, tpr, _ = metrics.roc_curve(y, prediction, sample_weight=w, pos_label=1)
    return metrics.auc(fpr, tpr, reorder=reorder)


def nDCG_cls(estimator, X, y, at=10, cnn_bdt=False, prediction=False):

    if prediction:
        prediction = estimator
    else:
        if cnn_bdt:
            prediction = estimator.predict_proba(X)[:,1]
        else:
            prediction = estimator.predict_proba(X)

    rand = np.random.random(len(prediction))
    sorted_list = [yi for _, _, yi in sorted(zip(prediction, rand, y), reverse=True)]
    yref = sorted(y, reverse=True)

    DCG = 0.
    IDCG = 0.
    for i in range(min(at, len(sorted_list))):
        DCG += (2 ** sorted_list[i] - 1) / np.log2(i + 2)
        IDCG += (2 ** yref[i] - 1) / np.log2(i + 2)
    if IDCG == 0:
        return 0
    nDCG = DCG / IDCG
    return nDCG


def comparator(x1, x2, estimator):
    """
    :param x1: list of documents
    :param x2: list of documents
    :return: cmp value for sorting the query
    """
    res = estimator.model.predict(
        [np.array([x1[:-1]]), np.array([x2[:-1]])],
        verbose=0
    )
    if res < 0:
        return -1
    elif res > 0:
        return 1
    return 0


def cmp_to_key(mycmp):
    """
    Convert a cmp= function into a key= function
    """

    class K:
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0

    return K


def readData(path=None, binary=True, at=10, number_features=136, cut=1.5, synth_data=False):
    x = []
    y = []
    q = []
    for line in open(path):
        s = line.split()
        if binary:
            if int(s[0]) > cut:
                y.append(1)
            else:
                y.append(0)
        else:
            y.append(int(s[0]))

        q.append(int(s[1].split(":")[1]))

        x.append(np.zeros(number_features))
        for i in range(number_features):
            x[-1][i] = float(s[i + 2].split(":")[1])

    x = np.array(x)
    y = np.array(y)
    q = np.array(q)

    if synth_data:
        return x, y, q

    xt = []
    yt = []
    qt = []

    for qid in np.unique(q):
        if len(y[q == qid]) < at: continue
        xt.extend(x[q == qid].tolist())
        yt.extend(y[q == qid].tolist())
        qt.extend(q[q == qid].tolist())

    return np.array(xt), np.array(yt), np.array(qt)


def readDataV1(data_path=None, debug_data=False, binary=False, preprocessing="quantile_3", at=10, number_features=136,
             bin_cutoff=1.5, cut_zeros=False, synth_data=False):
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

    if synth_data:
        return x, y, q

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


def AvgP_cls(estimator, X, y, prediction=False):
    if len(np.unique(y)) <= 1:
        return 0
    if prediction:
        avgp = average_precision_score(y, estimator)
    else:
        avgp = average_precision_score(y, estimator.predict_proba(X))
    #print('AvgP: ' + str(avgp))
    return avgp


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

    return float(np.mean(listOfAvgP))


def dcg_at_k(r, k, method=0):
	r = np.asfarray(r)[:k]
	if r.size:
		if method == 0:
			return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
		elif method == 1:
			return np.sum(r / np.log2(np.arange(2, r.size + 2)))
		else:
			raise ValueError('method must be 0 or 1.')
	return 0.


def ndcg_at_k(r, k, method=0):
	dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
	if not dcg_max:
		return 0.
	return dcg_at_k(r, k, method) / dcg_max


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
        listOfnDCG.append(nDCG_cls(estimator, query, y_query, k))
    return float(np.mean(listOfnDCG))


def train_model(num_features, path="fig2", test=False):
    x_train, y_train, _ = readData(
        path=f"{path}/train",
        binary=False,
        at=20,
        number_features=num_features,
        synth_data=True
    )

    x_test, y_test, q_test = readData(
        path=f"{path}/test",
        binary=False,
        at=20,
        number_features=num_features,
        synth_data=True
    )

    scaler = QuantileTransformer(output_distribution="normal")
    x_train = scaler.fit_transform(x_train) / 3
    x_test = scaler.transform(x_test) / 3

    print("Train Model")
    if test:
        dr = DirectRanker(
            hidden_layers_dr=[70, 5],
            num_features=num_features,
            epoch=1,
            verbose=2
        )
    else:
        dr = DirectRanker(
            hidden_layers_dr=[256, 128, 64, 20],
            drop_out=0.2,
            scale_factor_train_sample=3,
            num_features=num_features,
            epoch=20,
            verbose=2
        )
    dr.fit(x_train, y_train)

    print("Test Model")
    nDCG_l = []
    pred = dr.predict_proba(x_test)
    for n in np.unique(q_test):
        ndcg = nDCG_cls(pred[q_test==n], x_test[q_test==n], y_test[q_test==n], at=20, prediction=True)
        nDCG_l.append(ndcg)
    print(f"NDCG@20: {np.mean(nDCG_l)}")
    return np.mean(nDCG_l), np.std(nDCG_l)
