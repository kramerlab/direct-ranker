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

    ndcg_mean_70 = []
    ndcg_err_70 = []
    ndcg_mean_40 = []
    ndcg_err_40 = []
    for classes in np.arange(2, 20, 1):
        gen_set(n_feats=40, n_relevance=classes, path="fig2a", test=test)
        mean, err = train_model(num_features=40, path="fig2a", test=test)
        ndcg_mean_40.append(mean)
        ndcg_err_40.append(err)

        gen_set(n_feats=70, n_relevance=classes, path="fig2a", test=test)
        mean, err = train_model(num_features=70, path="fig2a", test=test)
        ndcg_mean_70.append(mean)
        ndcg_err_70.append(err)

    plt.errorbar(np.arange(2, 20, 1), ndcg_mean_40, yerr=ndcg_err_40, alpha=0.3)
    plt.scatter(np.arange(2, 20, 1), ndcg_mean_40, label="# Features 40")
    plt.errorbar(np.arange(2, 20, 1), ndcg_mean_70, yerr=ndcg_err_70, alpha=0.3)
    plt.scatter(np.arange(2, 20, 1), ndcg_mean_70, label="# Features 70")
    plt.xlabel(r'#Classes')
    plt.ylabel(r'nDCG@20')
    plt.savefig("fig2a.pdf")
    plt.close()
