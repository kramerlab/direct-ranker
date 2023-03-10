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
    n_trains = [int(i) for i in 10 ** np.arange(0, 7, 0.5) - 1 + 100]
    for n_train in n_trains:
        gen_set(n_train=n_train, n_feats=40, n_relevance=5, path="fig2a", test=test)
        mean, err = train_model(num_features=40, path="fig2a", test=test)
        ndcg_mean_40.append(mean)
        ndcg_err_40.append(err)

        gen_set(n_train=n_train, n_feats=70, n_relevance=5, path="fig2a", test=test)
        mean, err = train_model(num_features=70, path="fig2a", test=test)
        ndcg_mean_70.append(mean)
        ndcg_err_70.append(err)

    plt.errorbar(n_trains, ndcg_mean_40, yerr=ndcg_err_40, alpha=0.3)
    plt.scatter(n_trains, ndcg_mean_40, label="# Features 40")
    plt.errorbar(n_trains, ndcg_mean_70, yerr=ndcg_err_70, alpha=0.3)
    plt.scatter(n_trains, ndcg_mean_70, label="# Features 70")
    plt.xlabel(r'#Documents in training')
    plt.semilogx()
    plt.ylabel(r'nDCG@20')
    plt.savefig("fig2d.pdf")
    plt.close()
