import os
import argparse

from Rankers.helpers import readData, gen_set, comparator, cmp_to_key
from Rankers.models.DirectRanker import DirectRanker

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from collections import defaultdict
from functools import partial

import numpy as np
from sklearn.utils import shuffle


def plot(rankerValues, real_y, tuple_y, name, numberClass):

    colors = shuffle(cm.rainbow(np.linspace(0, 1, len(np.unique((real_y))))))
    dict_of_idx = defaultdict(list)
    dict_of_y_values = defaultdict(list)

    for idx, y in enumerate(tuple_y):
        dict_of_y_values[y[0]].append(rankerValues[idx])
        dict_of_y_values[y[1]].append(rankerValues[idx])
        dict_of_idx[y[0]].append(idx)
        dict_of_idx[y[1]].append(idx + 1)

    for k in sorted(dict_of_y_values.keys()):
        plt.scatter(dict_of_idx[k], dict_of_y_values[k], color=colors[int(k)])  # , edgecolors='black')
    plt.legend(np.unique(real_y), loc=1, title="Class labels")

    plt.xlabel(r'$n$')
    plt.ylabel(r'$r(d_n,d_{n+1})$')
    plt.savefig(name + ".pdf")
    plt.close()


def plot_sorted_list(estimator, X, y):
    X = np.array(X)
    y = np.array([[yi] for yi in y])

    data = np.concatenate((X, y), axis=1)
    compar = partial(comparator, estimator=estimator)
    data = np.array(sorted(data, key=cmp_to_key(compar), reverse=True))

    plot_values = []
    real_y = []
    tuple_y = []
    for idx, _ in enumerate(data):
        if idx == len(data) - 1:
            break
        res = estimator.model.predict(
            [np.array([data[idx][:-1]]), np.array([data[idx+1][:-1]])],
            verbose=0
        )
        plot_values.append(res[0][0])
        tuple_y.append((data[idx][-1], data[idx + 1][-1]))
        real_y.append(int(data[idx][-1]))
        real_y.append(int(data[idx + 1][-1]))

    return plot_values, real_y, tuple_y


def plot_own_data(path, n_relevance=3, test=False):
    gen_set(path=path, n_feats=136, n_relevance=n_relevance, test=test)

    x_train, y_train, _ = readData(path='output/train', binary=False, at=0)
    x_test, y_test, _ = readData(path='output/test', binary=False, at=0)

    # Load directRanker
    if test:
        dr = DirectRanker(epoch=1, num_features=x_train.shape[1])
    else:
        dr = DirectRanker(hidden_layers_dr=[50, 5], num_features=x_train.shape[1], verbose=2)

    dr.fit(x_train, y_train)

    plot_values, real_y, tuple_y = plot_sorted_list(dr, x_test, y_test)
    plot(plot_values, real_y, tuple_y, "own_data_class_" + str(n_relevance), n_relevance)


def plot_mslr(path, test=False, multi=False):

    if multi:
        x_train, y_train, _ = readData(path=f'{path}/Fold1/train.txt', binary=False, at=0)
        x_test, y_test, _ = readData(path=f'{path}/Fold1/test.txt', binary=False, at=0)
    else:
        x_train, y_train, _ = readData(path=f'{path}/Fold1/train.txt', binary=True, at=0)
        x_test, y_test, _ = readData(path=f'{path}/Fold1/test.txt', binary=False, at=0)

    # Load directRanker
    if test:
        dr = DirectRanker(epoch=1, num_features=x_train.shape[1])
    else:
        dr = DirectRanker(hidden_layers_dr=[50, 5], num_features=x_train.shape[1], verbose=2)

    dr.fit(x_train, y_train)

    plot_values, real_y, tuple_y = plot_sorted_list(dr, x_test, y_test)
    if multi:
        plot(plot_values, real_y, tuple_y, "mslr_multi_" + str(multi), 5)
    else:
        plot(plot_values, real_y, tuple_y, "mslr_multi_" + str(multi), 2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get some run values.')
    parser.add_argument('--test', type=int, default=0, help='Run in test mode')
    parser.add_argument('--data', type=str, default="", help='path to data')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not os.path.exists("output"):
        os.makedirs("output")

    test = False
    if args.test == 1:
        test = True

    print("Run own generated data test")
    plot_own_data("output", n_relevance=3, test=test)

    print("Run MSLR multi label")
    plot_mslr(args.data, test=test, multi=True)

    print("Run MSLR binary label")
    plot_mslr(args.data, test=test, multi=False)
