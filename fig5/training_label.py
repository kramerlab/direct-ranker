import os

from Rankers.models.DirectRanker import DirectRanker
from Rankers.models.RankNet import RankNet
from Rankers.models.ListNet import ListNet
from Rankers.helpers import nDCG_cls
from Rankers.models.LambdaRank import LambdaRank

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import shuffle
from tqdm import tqdm
from time import time

import json
import random
import argparse


def gen_set(n_relevance=6, n_train=100000, n_test=10000, overwrite=False, test=False):

    for fold in range(5):
        if os.path.isfile(data_path + "/train_label_{}_{}_{}".format(fold, n_train, n_test)):
            if overwrite:
                os.remove(data_path + "/train_label_{}_{}_{}".format(fold, n_train, n_test)) 
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
        n_relevance = n_relevance
        means = np.random.rand(n_relevance, n_feats) * 100
        sigmas = np.random.rand(n_relevance, n_feats) * 150 + 50

        f = open(data_path + "/train_label_{}_{}_{}".format(fold, n_train, n_test),"w")
        print("Creating traing set...")
        for i in tqdm(range(n_train // n_relevance)):
            for j in range(n_relevance):
                f.write(str(j))
                #f.write("\t" + str(int(random.uniform(0, max_qid_train))))
                for n in np.random.randn(n_feats)*sigmas[j]+means[j]:
                    f.write("\t"+str(n))
                f.write("\n")
        f.close()

        f = open(data_path + "/test_label_{}_{}_{}".format(fold, n_train, n_test),"w")
        print("Creating test set...")
        for i in tqdm(range(n_test // n_relevance)):
            for j in range(n_relevance):
                f.write(str(j))
                #f.write("\t" + str(int(random.uniform(0, max_qid_test))))
                for n in np.random.randn(n_feats)*sigmas[j]+means[j]:
                    f.write("\t"+str(n))
                f.write("\n")
        f.close()

def test_rankers(n_relevance=6, n_train=100000, n_test=10000, overwrite=False, add_model=None, test=False):
    
    if test:
        data_str = "100_10"
        folds = 1
    else:
        data_str = "100000_10000"
        folds = 5

    if add_model == None:
        results = {"n_train": n_train, "n_test": n_test}
    ndcg_dr_l = []
    ndcg_lnet_l = []
    ndcg_rnet_l = []
    ndcg_lrank_l = []
    for fold in range(folds):
        if os.path.isfile(results_path + "/results_label_{}_{}.json".format(n_train, n_test)):
            if not (add_model == None):
                results = json.load(open(results_path + "/results_label_{}_{}.json".format(n_train, n_test)))
            elif overwrite:
                os.remove(results_path + "/results_label_{}_{}.json".format(n_train, n_test)) 
            else:
                print("skip training for n_relevance {}".format(n_train))
                return 0
        
        size_min = 50
        size_max = 150
        N = 50
        x = []
        y = []
        x_test = []
        y_test = []

        # read data
        print("Reading data")
        start = time()
        if os.path.isfile(data_path + "/x_train_label_{}.npy".format(fold)):
            x = np.load(data_path + "/x_train_label_{}.npy".format(fold))
            y = np.load(data_path + "/y_train_label_{}.npy".format(fold))
            x_test = np.load(data_path + "/x_test_label_{}.npy".format(fold))
            y_test = np.load(data_path + "/y_test_label_{}.npy".format(fold))
        else:
            for line in open(data_path + "/train_label_{}_{}".format(fold, data_str)):
                s = line.split()
                y.append(int(str(s[0])))
                x.append(np.zeros(len(s) - 1))
                for i in range(len(s) - 1):
                    x[-1][i] = float(s[i + 1])

            for line in open(data_path + "/test_label_{}_{}".format(fold, data_str)):
                s = line.split()
                y_test.append(int(str(s[0])))
                x_test.append(np.zeros(len(s) - 1))
                for i in range(len(s) - 1):
                    x_test[-1][i] = float(s[i + 1])

            end = time()
            mins = int((end - start) / 60)
            secs = end - start - 60 * mins
            print("Finished in " + str(mins) + " min " + str(secs) + " s")
            x = np.array(x)
            y = np.array(y)
            x, y = shuffle(x, y)
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            x_test, y_test = shuffle(x_test, y_test)
            np.save(data_path + "/x_train_label_{}.npy".format(fold), x)
            np.save(data_path + "/y_train_label_{}.npy".format(fold), y)
            np.save(data_path + "/x_test_label_{}.npy".format(fold), x_test)
            np.save(data_path + "/y_test_label_{}.npy".format(fold), y_test)

        print("Preparing data n_train {} on fold {}".format(n_train, fold))
        # draw fartion
        if n_train >= 1e6: return
        # TODO: make me dynamic 
        x5 = x[y==5]
        x4 = x[y==4]
        x3 = x[y==3]
        x0 = x[y<3]
        y0 = y[y<3]
        if n_train < 50000:
            x5 = np.array(random.sample(x5.tolist(), int(n_train/3)))
            x4 = np.array(random.sample(x4.tolist(), int(n_train/3)))
            x3 = np.array(random.sample(x3.tolist(), int(n_train/3)))
        x = np.concatenate([x0, x3, x4, x5])
        y = np.concatenate(
            [
                y0,
                np.ones(len(x3))*3,
                np.ones(len(x4))*4,
                np.ones(len(x5))*5
            ]).astype(int)
        x, y = shuffle(x, y)
        y_listnet = []
        for yi in y:
            if yi > (n_relevance-1)/2:
                y_listnet.append(1)
            else:
                y_listnet.append(0)

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
            dr.fit(x, y)
        if not ('ndcg_RankNet' in results) or add_model == 'RankNet':
            ranknet.fit(x, y)
        if not ('ndcg_ListNet' in results) or add_model == 'ListNet':
            lnet.fit(x, tf.cast(y_listnet, tf.float32))
        if not('ndcg_LambdaRank' in results) or add_model == 'LambdaRank':
            lrank.fit(x, y)
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

    with open(results_path + '/results_label_{}_{}.json'.format(n_train, n_test), 'w', encoding='utf-8') as f:
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

    train_list = np.logspace(2, 6, num=10)
    train_list = np.concatenate([np.logspace(1, 2, num=3)[:2], train_list])
    for n_train in train_list:
        gen_set(overwrite=False, test=test)
        test_rankers(n_train=int(n_train), n_test=10000, overwrite=True, test=test)
        if test: break
