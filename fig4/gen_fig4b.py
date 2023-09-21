import os

from Rankers.helpers import gen_set, train_model

import matplotlib.pyplot as plt
import numpy as np

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get some run values.')
    parser.add_argument('--test', type=int, default=0, help='Run in test mode')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    test = False
    if args.test == 1:
        test = True

    ndcg_mean_5 = []
    ndcg_err_5 = []
    ndcg_mean_10 = []
    ndcg_err_10 = []
    for features in np.arange(2, 100, 5):
        gen_set(n_feats=features, n_relevance=5, path="fig2b", test=test)
        mean, err = train_model(num_features=int(features), path="fig2b", test=test)
        ndcg_mean_5.append(mean)
        ndcg_err_5.append(err)

        gen_set(n_feats=features, n_relevance=10, path="fig2b", test=test)
        mean, err = train_model(num_features=int(features), path="fig2b", test=test)
        ndcg_mean_10.append(mean)
        ndcg_err_10.append(err)

    plt.errorbar(np.arange(2, 100, 5), ndcg_mean_5, yerr=ndcg_err_5, alpha=0.3)
    plt.scatter(np.arange(2, 100, 5), ndcg_mean_5, label="# Classes 5")
    plt.errorbar(np.arange(2, 100, 5), ndcg_mean_10, yerr=ndcg_err_10, alpha=0.3)
    plt.scatter(np.arange(2, 100, 5), ndcg_mean_10, label="# Classes 10")
    plt.xlabel(r'#Features')
    plt.ylabel(r'nDCG@20')
    plt.savefig("fig2b.pdf")
    plt.close()
