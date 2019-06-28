import sys

sys.path.append('..')

from supplementary_code_paper.helpers import nDCGScorer_cls, readData, MAP_cls
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from supplementary_code_paper.DirectRanker import directRanker
from functools import partial
import numpy as np

list_of_train = ["/MSLR-WEB10K/Fold1/train.txt", "/MSLR-WEB10K/Fold2/train.txt",
                 "/MSLR-WEB10K/Fold3/train.txt", "/MSLR-WEB10K/Fold4/train.txt",
                 "/MSLR-WEB10K/Fold5/train.txt"]
list_of_test = ["/MSLR-WEB10K/Fold1/test.txt", "/MSLR-WEB10K/Fold2/test.txt",
                "/MSLR-WEB10K/Fold3/test.txt", "/MSLR-WEB10K/Fold4/test.txt",
                "/MSLR-WEB10K/Fold5/test.txt"]

try_name = "MSLR_ndcg"

parameters = {
    'hidden_layers': [[10], [20], [30], [50], [100]],
    'weight_regularization': [0., 0.0001, 0.001],
    'early_stopping': [False],
    'dropout': [0., 0.5]
}

nDCGScorer10 = partial(nDCGScorer_cls, at=10)

scoring = {'NDGC@10': nDCGScorer10, 'MAP': MAP_cls}

counter_id = 0
counters = [1, 2, 3, 4, 5]
list_of_ndcg = []
list_of_map = []
for train, test in zip(list_of_train, list_of_test):
    counter = counters[counter_id]
    counter_id = counter_id + 1
    print("FOLD ", counter)

    # Load directRanker and set up gridsearch
    dr = directRanker(
        feature_activation=tf.nn.tanh,
        ranking_activation=tf.nn.tanh,
        max_steps=3000,
        print_step=0,
        start_batch_size=3,
        end_batch_size=10,
        start_qids=10,
        end_qids=100,
        weight_regularization=0.0,
        early_stopping=False,
        feature_bias=True,
        hidden_layers=[20],
        random_seed=42,
        dropout=0.0,
        name="mq_test"
    )
    # MSLR binarize 1 if label > 1.5 else 0
    x_train, y_train, _ = readData(data_path=train, binary=True, at=10, number_features=136, bin_cutoff=1.5,
                                   cut_zeros=True)
    x_test, y_test, _ = readData(data_path=test, binary=True, at=10, number_features=136, bin_cutoff=1.5,
                                 cut_zeros=True)

    clf = GridSearchCV(dr, parameters, cv=5, n_jobs=4, verbose=1, scoring=scoring, refit='NDGC@10',
                       return_train_score=False)
    clf.fit(x_train, y_train, ranking=True)
    best_estimator = clf.best_estimator_

    print("Best Parameters:")
    print(clf.best_params_)

    score_map = MAP_cls(best_estimator, x_test, y_test)
    score = nDCGScorer10(best_estimator, x_test, y_test)

    print("Test on Fold" + str(counter) + ": NDCG@10=" + str(score) + "  MAP=" + str(score_map))

    list_of_ndcg.append(score)
    list_of_map.append(score_map)
    np.save(try_name + "_direct_ranker_results_gridsearch" + str(counter), clf.cv_results_)

np.save("{}_list_of_ndcg".format(try_name), list_of_ndcg)
np.save("{}_list_of_map".format(try_name), list_of_map)
