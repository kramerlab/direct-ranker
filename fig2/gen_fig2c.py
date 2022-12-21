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

    ndcg_mean = []
    ndcg_err = []
    for sigma in np.arange(0, 1.1, 0.1):
        gen_set(n_feats=70, n_relevance=5, sigma_label=sigma, path="fig2c", test=test)
        mean, err = train_model(num_features=70, path="fig2c", test=test)
        ndcg_mean.append(mean)
        ndcg_err.append(err)

    plt.errorbar(np.arange(0, 1.1, 0.1), ndcg_mean, yerr=ndcg_err, alpha=0.3)
    plt.scatter(np.arange(0, 1.1, 0.1), ndcg_mean)
    plt.xlabel(r'$\sigma$')
    plt.ylabel(r'nDCG@20')
    plt.savefig("fig2c.pdf")
    plt.close()
