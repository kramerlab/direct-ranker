from __future__ import print_function

import sys
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

sys.path.append('..')
import tensorflow as tf
from supplementary_code_direct_ranker.DirectRanker import directRanker
from supplementary_code_direct_ranker.helpers import readData, nDCGScorer_cls, MAP_cls

# CHANGE THE DATA PATH!!!
# Change number_features to 46 and binary=False for MQ2007/8 and 136 and binary=True for MSLR
x_train, y_train, q_train = readData(data_path="../data/MSLR-WEB10K/Fold1/train.txt", binary=True, at=10, number_features=136, bin_cutoff=1.5, cut_zeros=True)
# For debugging
# x_train, y_train, q_train = readData(debug_data=True, binary=True, at=10, number_features=136, bin_cutoff=1.5,
#                                      cut_zeros=True)
x_test, y_test, q_test = readData(data_path="../data/MSLR-WEB10K/Fold1/test.txt", binary=True, at=10, number_features=136, bin_cutoff=1.5, cut_zeros=True)
# For debugging
# x_test, y_test, q_test = readData(debug_data=True, binary=True, at=10, number_features=136, bin_cutoff=1.5,
#                                   cut_zeros=True)

x = np.load("../data/MSLR-WEB10K/x_total.npy")
x = QuantileTransformer(output_distribution="normal").fit_transform(x) / 3
#x = StandardScaler().fit_transform(x)
# y = np.array([np.load("../data/MSLR-WEB10K/y_total.npy")]).transpose()

y = np.load("../data/MSLR-WEB10K/y_total.npy")
for idx, v in enumerate(y):
    if v > 1.5:
        y[idx] = 1
    else:
        y[idx] = 0
y = np.array([y]).transpose()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

def lambda_cost(nn, y0):
    return tf.reduce_mean(tf.log(1+tf.exp((1+nn)/2))-(1+nn)/2)


# Load directRanker, train, and test
dr = directRanker(
    feature_activation=tf.nn.tanh,
    ranking_activation=tf.nn.tanh,
    # max_steps=10000,
    # For debugging
    cost=lambda_cost,
    max_steps=5000,
    print_step=500,
    start_batch_size=3,
    end_batch_size=5,
    start_qids=20,
    end_qids=100,
    feature_bias=True,
    hidden_layers=[30, 5]
)

dr.fit(x_train, y_train, ranking=True)

nDCGScorer_cls(dr, x_test, y_test, at=10)
MAP_cls(dr, x_test, y_test)
