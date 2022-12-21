import os
import tensorflow as tf

from Rankers.models.DirectRanker import DirectRanker
from Rankers.models.RankNet import RankNet
from Rankers.models.ListNet import ListNet
from Rankers.helpers import nDCG_cls
from Rankers.models.LambdaRank import LambdaRank

import numpy as np
from multiprocessing import Pool
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import shuffle
from datetime import datetime
from tqdm import tqdm
import argparse
from time import time

import json
import random
import argparse


def gen_set(n_relevance=5, n_train=100000, n_test=10000, overwrite=False, test=False):

    for fold in range(5):
        if os.path.isfile(data_path + "/train_{}_{}_{}".format(fold, n_train, n_test)):
            if overwrite:
                os.remove(data_path + "/train_{}_{}_{}".format(fold, n_train, n_test)) 
            else:
                print("skip n_train file {}".format(n_train))
                return 0
        n_feats = 70
        if test:
            n_train = 100
            n_test = 10
        else:
            n_train = n_train
            n_test = n_test
        n_relevance = 5
        max_qid_train = 10
        max_qid_test = 20
        means = np.random.rand(n_relevance, n_feats) * 100
        sigmas = np.random.rand(n_relevance, n_feats) * 150 + 50

        f = open(data_path + "/train_{}_{}_{}".format(fold, n_train, n_test),"w")
        print("Creating traing set...")
        for i in tqdm(range(n_train // n_relevance)):
            for j in range(n_relevance):
                f.write(str(j))
                #f.write("\t" + str(int(random.uniform(0, max_qid_train))))
                for n in np.random.randn(n_feats)*sigmas[j]+means[j]:
                    f.write("\t"+str(n))
                f.write("\n")
        f.close()

        f = open(data_path + "/test_{}_{}_{}".format(fold, n_train, n_test),"w")
        print("Creating test set...")
        for i in tqdm(range(n_test // n_relevance)):
            for j in range(n_relevance):
                f.write(str(j))
                #f.write("\t" + str(int(random.uniform(0, max_qid_test))))
                for n in np.random.randn(n_feats)*sigmas[j]+means[j]:
                    f.write("\t"+str(n))
                f.write("\n")
        f.close()

def test_rankers(n_relevance=5, n_train=100000, n_test=10000, overwrite=False, add_model=None, test=False):
    
    folds = 5
    if test:
        n_train = 100
        n_test = 10
        folds = 1

    if add_model == None:
        results = {"n_train": n_train, "n_test": n_test}
    ndcg_dr_l = []
    ndcg_lnet_l = []
    ndcg_rnet_l = []
    ndcg_lrank_l = []
    for fold in range(folds):
        if os.path.isfile(results_path + "/results_{}_{}.json".format(n_train, n_test)):
            if not (add_model == None):
                results = json.load(open(results_path + "/results_{}_{}.json".format(n_train, n_test)))
            elif overwrite:
                os.remove(results_path + "/results_{}_{}.json".format(n_train, n_test)) 
            else:
                print("skip training for n_relevance {}".format(n_train))
                return 0
        
        size_min = 50
        size_max = 150
        N = 50
        x = []
        y = []
        y_dr = []
        x_test = []
        y_test = []

        # read data
        print("Reading data")
        start = time()
        for line in open(data_path + "/train_{}_{}_{}".format(fold, n_train, n_test)):
            s = line.split()
            if int(s[0]) > (n_relevance-1)/2:
                y.append(1)
            else:
                y.append(0)
            y_dr.append(int(s[0]))
            x.append(np.zeros(len(s) - 1))
            for i in range(len(s) - 1):
                x[-1][i] = float(s[i + 1])

        for line in open(data_path + "/test_{}_{}_{}".format(fold, n_train, n_test)):
            s = line.split()
            y_test.append(int(str(s[0])))
            x_test.append(np.zeros(len(s) - 1))
            for i in range(len(s) - 1):
                x_test[-1][i] = float(s[i + 1])
                
        end = time()
        mins = int((end - start) / 60)
        secs = end - start - 60 * mins
        print("Finished in " + str(mins) + " min " + str(secs) + " s")

        print("Preparing data n_train {} on fold {}".format(n_train, fold))
        x = np.array(x)
        y = np.array(y)
        y_dr = np.array(y_dr)
        x, y, y_dr = shuffle(x, y, y_dr)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        x_test, y_test = shuffle(x_test, y_test)
        
        scaler = QuantileTransformer(output_distribution="normal")
        x = scaler.fit_transform(x) / 3
        x_test = scaler.transform(x_test) / 3

        if test:
            epoch = 1
        else:
            epoch = 10

        dr = DirectRanker(
            hidden_layers_dr=[70, 5], 
            num_features=len(x[0]), 
            verbose=0, 
            epoch=epoch,
            scale_factor_train_sample=1
        )
        
        ranknet = RankNet(
            hidden_layers_dr=[70, 5], 
            num_features=len(x[0]), 
            verbose=0,
            epoch=epoch,
            scale_factor_train_sample=1
        )
        
        lnet = ListNet(
            hidden_layers_dr=[70, 5, 1], 
            num_features=len(x[0]), 
            verbose=0, 
            epoch=epoch
        )

        lrank = LambdaRank(
            hidden_layers_dr=[70, 5],
            num_features=len(x[0]),
            verbose=2, 
            epoch=epoch,
            scale_factor_train_sample=1
            )

        print("Training")
        start = time()
        if not ('ndcg_DirectRanker' in results) or add_model == 'DirectRanker':
            dr.fit(x, y_dr)
        if not ('ndcg_RankNet' in results) or add_model == 'RankNet':
            ranknet.fit(x, y_dr)
        if not ('ndcg_ListNet' in results) or add_model == 'ListNet':
            lnet.fit(x, tf.cast(y, tf.float32))
        if not('ndcg_LambdaRank' in results) or add_model == 'LambdaRank':
            lrank.fit(x, y_dr)
        end = time()
        mins = int((end - start) / 60)
        secs = end - start - 60 * mins
        print("Finished in " + str(mins) + " min " + str(secs) + " s")
        ndcg_dr_l_cur = []
        ndcg_lnet_l_cur = []
        ndcg_rnet_l_cur = []
        nDCG_lrank_l_cur = []
        for n in range(N):
            size = np.random.randint(size_min,size_max)
            ind = np.random.randint(0,len(x_test),size)
            
            if not ('ndcg_DirectRanker' in results) or add_model == 'DirectRanker':
                nDCG_dr = nDCG_cls(dr, x_test[ind], y_test[ind], at=20)
                print("nDCG@20 DirectRanker {} fold {} n_train {}".format(nDCG_dr, fold, n_train))
            if not ('ndcg_RankNet' in results) or add_model == 'RankNet':
                nDCG_rnet = nDCG_cls(ranknet, x_test[ind], y_test[ind], at=20)
                print("nDCG@20 RankNet {} fold {} n_train {}".format(nDCG_rnet, fold, n_train))
            if not ('ndcg_ListNet' in results) or add_model == 'ListNet':
                nDCG_lnet = nDCG_cls(lnet, x_test[ind], y_test[ind], at=20)
                print("nDCG@20 ListNet {} fold {} n_train {}".format(nDCG_lnet, fold, n_train))
            if not ('ndcg_LambdaRank' in results) or add_model == 'LambdaRank':
                nDCG_lrank = nDCG_cls(lrank, x_test[ind], y_test[ind], at=20)
                print("nDCG@20 LambdaRank {} fold {} n_train {}".format(nDCG_lrank, fold, n_train))

            if not ('ndcg_DirectRanker' in results) or add_model == 'DirectRanker':
                ndcg_dr_l_cur.append(nDCG_dr)
            if not ('ndcg_RankNet' in results) or add_model == 'RankNet':
                ndcg_rnet_l_cur.append(nDCG_rnet)
            if not ('ndcg_ListNet' in results) or add_model == 'ListNet':
                ndcg_lnet_l_cur.append(nDCG_lnet)
            if not ('ndcg_LambdaRank' in results) or add_model == 'LambdaRank':
                nDCG_lrank_l_cur.append(nDCG_lrank)
        
        ndcg_dr_l.append(np.mean(ndcg_dr_l_cur))
        ndcg_lnet_l.append(np.mean(ndcg_lnet_l_cur))
        ndcg_rnet_l.append(np.mean(ndcg_rnet_l_cur))
        ndcg_lrank_l.append(np.mean(nDCG_lrank_l_cur))
    
    if not ('ndcg_DirectRanker' in results) or add_model == 'DirectRanker':
        results['ndcg_DirectRanker'] = [np.mean(ndcg_dr_l), np.std(ndcg_dr_l)]
    if not ('ndcg_RankNet' in results) or add_model == 'RankNet':
        results['ndcg_RankNet'] = [np.mean(ndcg_rnet_l), np.std(ndcg_rnet_l)]
    if not ('ndcg_ListNet' in results) or add_model == 'ListNet':
        results['ndcg_ListNet'] = [np.mean(ndcg_lnet_l), np.std(ndcg_lnet_l)]
    if not ('ndcg_LambdaRank' in results) or add_model == 'LambdaRank':
        results['ndcg_LambdaRank'] = [np.mean(ndcg_lrank_l), np.std(ndcg_lrank_l)]

    # hack to be able to plot the results in test mode
    if test:
        n_test = 10000

    with open(results_path + '/results_{}_{}.json'.format(n_train, n_test), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get some run values.')
    parser.add_argument('--test', type=int, default=0, help='Run in test mode')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not os.path.exists("output"):
        os.makedirs("output")

    test = False
    if args.test == 1:
        test = True

    path = 'synth_data'
    data_path = path + "/data"
    results_path = path + "/results"

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for n_train in np.logspace(2, 6, num=10):
        gen_set(n_train=int(n_train), n_test=10000, overwrite=True, test=test)
        test_rankers(n_train=int(n_train), n_test=10000, overwrite=True, test=test)
        if test: break
