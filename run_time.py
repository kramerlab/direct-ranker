from __future__ import print_function

import sys
import numpy as np

sys.path.append('..')
import tensorflow as tf
from supplementary_code_direct_ranker.DirectRanker import directRanker
from supplementary_code_direct_ranker.helpers import readData
import timeit

# CHANGE THE DATA PATH!!!
# Change number_features to 46 and binary=False for MQ2007/8 and 136 and binary=True for MSLR
# x_train, y_train, q_train = readData(data_path="../data/MSLR-WEB10K/Fold1/train.txt", binary=True, at=10, number_features=136, bin_cutoff=1.5, cut_zeros=True)
# For debugging
x_train, y_train, q_train = readData(debug_data=True, binary=True, at=10, number_features=136, bin_cutoff=1.5,
                                     cut_zeros=True)
# x_test, y_test, q_test = readData(data_path="../data/MSLR-WEB10K/Fold1/test.txt", binary=True, at=10, number_features=136, bin_cutoff=1.5, cut_zeros=True)
# For debugging
x_test, y_test, q_test = readData(debug_data=True, binary=True, at=10, number_features=136, bin_cutoff=1.5,
                                  cut_zeros=True)

time_dr = []
time_rank = []

def ranknet_cost(nn, y0):
    return tf.reduce_mean(tf.log(1+tf.exp((1+nn)/2))-(1+nn)/2)

for i in range(10):

    # Load directRanker, train, and test
    dr = directRanker(
        feature_activation=tf.nn.tanh,
        ranking_activation=tf.nn.tanh,
        # max_steps=10000,
        # For debugging
        max_steps=5000,
        print_step=500,
        start_batch_size=3,
        end_batch_size=5,
        start_qids=20,
        end_qids=100,
        feature_bias=True,
        hidden_layers=[30, 5]
    )

    ranknet = directRanker(
        feature_activation=tf.nn.relu,
        ranking_activation=tf.nn.tanh,
        # max_steps=10000,
        # For debugging
        max_steps=5000,
        print_step=500,
        optimizer=tf.train.GradientDescentOptimizer,
        start_batch_size=3,
        end_batch_size=5,
        start_qids=20,
        end_qids=100,
        feature_bias=True,
        cost=ranknet_cost,
        hidden_layers=[30, 5]
    )

    start = timeit.default_timer()
    ranknet.fit(x_train, y_train, ranking=True)
    stop = timeit.default_timer()
    time_rank.append(stop - start)
    print('Time RankNet: ', stop - start)

    start = timeit.default_timer()
    dr.fit(x_train, y_train, ranking=True)
    stop = timeit.default_timer()
    time_dr.append(stop - start)
    print('Time DirectRanker: ', stop - start)

print('Mean Time DirectRanker: ' + np.mean(time_dr) + " +- " + np.std(time_dr))
print('Mean Time RankNet: ' + np.mean(time_rank) + " +- " + np.std(time_rank))