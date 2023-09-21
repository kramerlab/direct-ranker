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


def readData(path=None, binary=True, at=10):
    x = []
    y = []
    q = []
    for line in open(path):
        s = line.split()
        if binary:
            if int(s[0]) > 1.5:
                y.append(1)
            else:
                y.append(0)
        else:
            y.append(int(s[0]))

        q.append(int(s[1].split(":")[1]))

        x.append(np.zeros(136))
        for i in range(136):
            x[-1][i] = float(s[i + 2].split(":")[1])

    x = np.array(x)
    y = np.array(y)
    q = np.array(q)

    xt = []
    yt = []
    qt = []

    for qid in np.unique(q):
        if len(y[q == qid]) < at: continue
        xt.extend(x[q == qid].tolist())
        yt.extend(y[q == qid].tolist())
        qt.extend(q[q == qid].tolist())

    return np.array(xt), np.array(yt), np.array(qt)

def test_rankers(n_train=100000, overwrite=False, add_model=None, test=False):

    folds = 5
    if test:
        folds = 1
    
    if add_model == None:
        results = {"n_train": n_train}
    ndcg_dr_l = []
    ndcg_lnet_l = []
    ndcg_rnet_l = []
    ndcg_lrank_l = []
    for fold in range(folds):
        if os.path.isfile(results_path + "/results_label_{}.json".format(n_train)):
            if not (add_model == None):
                results = json.load(open(results_path + "/results_label_{}.json".format(n_train)))
            elif overwrite:
                os.remove(results_path + "/results_label_{}.json".format(n_train)) 
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
        if os.path.isfile(output_path + "/data/x_train_label_fold{}.npy".format(fold+1)):
            x_train = np.load(output_path + "/data/x_train_label_fold{}.npy".format(fold+1))
            y_train = np.load(output_path + "/data/y_train_label_fold{}.npy".format(fold+1))
            
            x_test = np.load(output_path + "/data/x_test_label_fold{}.npy".format(fold+1))
            y_test = np.load(output_path + "/data/y_test_label_fold{}.npy".format(fold+1))
            q_test = np.load(output_path + "/data/q_test_label_fold{}.npy".format(fold+1))
        else:
            x_train, y_train, _ = readData(path=data_path + "/Fold{}/train.txt".format(fold+1), binary=False)
            x_test, y_test, q_test = readData(path=data_path + "/Fold{}/test.txt".format(fold+1), binary=False)
        
            np.save(output_path + "/data/x_train_label_fold{}".format(fold+1), x_train)
            np.save(output_path + "/data/y_train_label_fold{}".format(fold+1), y_train)
            
            np.save(output_path + "/data/x_test_label_fold{}".format(fold+1), x_test)
            np.save(output_path + "/data/y_test_label_fold{}".format(fold+1), y_test)
            np.save(output_path + "/data/q_test_label_fold{}".format(fold+1), q_test)
        end = time()
        mins = int((end - start) / 60)
        secs = end - start - 60 * mins
        print("Finished in " + str(mins) + " min " + str(secs) + " s")

        print("Preparing data n_train {} on fold {}".format(n_train, fold))
        # draw fartion
        if n_train >= 2e6: return
        # TODO: make me dynamic 
        x4 = x_train[y_train==4]
        x3 = x_train[y_train==3]
        x2 = x_train[y_train==2]
        x0 = x_train[y_train<2]
        y0 = y_train[y_train<2]
        total = len(y_train[y_train>=2])
        if n_train < 50000:
            x4 = np.array(random.sample(x4.tolist(), int(n_train*len(x4)/total)))
            x3 = np.array(random.sample(x3.tolist(), int(n_train*len(x3)/total)))
            x2 = np.array(random.sample(x2.tolist(), int(n_train*len(x2)/total)))
        x_list = []
        y_list = []
        for idx, xi in enumerate([x0, x2, x3, x4]):
            if len(xi) > 0:
                x_list.append(xi)
                y_list.append(np.ones(len(xi))*idx)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list).astype(int)
        x, y = shuffle(x, y)
        y_listnet = []
        for yi in y:
            if yi > 1.5:
                y_listnet.append(1)
            else:
                y_listnet.append(0)
        y_listnet = np.array(y_listnet)

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
            optimizer=tf.keras.optimizers.SGD,
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
            dr.fit(x, y_listnet)
        if not ('ndcg_RankNet' in results) or add_model == 'RankNet':
            ranknet.fit(x, y_listnet)
        if not ('ndcg_ListNet' in results) or add_model == 'ListNet':
            lnet.fit(x, tf.cast(y_listnet, tf.float32))
        if not('ndcg_LambdaRank' in results) or add_model == 'LambdaRank':
            lrank.fit(x, y_listnet)
        end = time()
        mins = int((end - start) / 60)
        secs = end - start - 60 * mins
        print("Finished in " + str(mins) + " min " + str(secs) + " s")
        ndcg_dr_l_cur = []
        ndcg_lnet_l_cur = []
        ndcg_rnet_l_cur = []
        nDCG_lrank_l_cur = []

        if not ('ndcg_DirectRanker' in results) or add_model == 'DirectRanker':
            dr_prediction = dr.predict_proba(x_test)
        if not ('ndcg_RankNet' in results) or add_model == 'RankNet':
            ranknet_prediction = ranknet.predict_proba(x_test)
        if not ('ndcg_ListNet' in results) or add_model == 'ListNet':
            lnet_prediction = np.array(lnet.predict_proba(x_test))
        if not ('ndcg_LambdaRank' in results) or add_model == 'LambdaRank':
            lrank_prediction = lrank.predict_proba(x_test)

        for n in np.unique(q_test):
            xt = x_test[q_test == n]
            yt = y_test[q_test == n]
            if not ('ndcg_DirectRanker' in results) or add_model == 'DirectRanker':
                nDCG_dr = nDCG_cls(dr_prediction[q_test == n], xt, yt, at=10, prediction=True)
            if not ('ndcg_RankNet' in results) or add_model == 'RankNet':
                nDCG_rnet = nDCG_cls(ranknet_prediction[q_test == n], xt, yt, at=10, prediction=True)
                #print("nDCG@10 RankNet {} fold {} n_train {}".format(nDCG_rnet, fold, n_train))
            if not ('ndcg_ListNet' in results) or add_model == 'ListNet':
                nDCG_lnet = nDCG_cls(lnet_prediction[q_test == n], xt, yt, at=10, prediction=True)
                #print("nDCG@10 ListNet {} fold {} n_train {}".format(nDCG_lnet, fold, n_train))
            if not ('ndcg_LambdaRank' in results) or add_model == 'LambdaRank':
                nDCG_lrank = nDCG_cls(lrank_prediction[q_test == n], xt, yt, at=10, prediction=True)
                #print("nDCG@10 LambdaRank {} fold {} n_train {}".format(nDCG_lrank, fold, n_train))

            if not ('ndcg_DirectRanker' in results) or add_model == 'DirectRanker':
                if str(nDCG_dr) != 'nan':
                    ndcg_dr_l_cur.append(nDCG_dr)
            if not ('ndcg_RankNet' in results) or add_model == 'RankNet':
                if str(nDCG_rnet) != 'nan':
                    ndcg_rnet_l_cur.append(nDCG_rnet)
            if not ('ndcg_ListNet' in results) or add_model == 'ListNet':
                if str(nDCG_lnet) != 'nan':
                    ndcg_lnet_l_cur.append(nDCG_lnet)
            if not ('ndcg_LambdaRank' in results) or add_model == 'LambdaRank':
                if str(nDCG_lrank) != 'nan':
                    nDCG_lrank_l_cur.append(nDCG_lrank)
        
        ndcg_dr_l.append(np.mean(ndcg_dr_l_cur))
        ndcg_lnet_l.append(np.mean(ndcg_lnet_l_cur))
        ndcg_rnet_l.append(np.mean(ndcg_rnet_l_cur))
        ndcg_lrank_l.append(np.mean(nDCG_lrank_l_cur))

        if not ('ndcg_DirectRanker' in results) or add_model == 'DirectRanker':
            print("nDCG@10 DirectRanker {} fold {} n_train {}".format(np.mean(ndcg_dr_l_cur), fold, n_train))
        if not ('ndcg_RankNet' in results) or add_model == 'RankNet':
            print("nDCG@10 RankNet {} fold {} n_train {}".format(np.mean(ndcg_rnet_l_cur), fold, n_train))
        if not ('ndcg_ListNet' in results) or add_model == 'ListNet':
            print("nDCG@10 ListNet {} fold {} n_train {}".format(np.mean(ndcg_lnet_l_cur), fold, n_train))
        if not ('ndcg_LambdaRank' in results) or add_model == 'LambdaRank':
            print("nDCG@10 LambdaRank {} fold {} n_train {}".format(np.mean(nDCG_lrank_l_cur), fold, n_train))
    
    if not ('ndcg_DirectRanker' in results) or add_model == 'DirectRanker':
        results['ndcg_DirectRanker'] = [np.mean(ndcg_dr_l), np.std(ndcg_dr_l)]
    if not ('ndcg_RankNet' in results) or add_model == 'RankNet':
        results['ndcg_RankNet'] = [np.mean(ndcg_rnet_l), np.std(ndcg_rnet_l)]
    if not ('ndcg_ListNet' in results) or add_model == 'ListNet':
        results['ndcg_ListNet'] = [np.mean(ndcg_lnet_l), np.std(ndcg_lnet_l)]
    if not ('ndcg_LambdaRank' in results) or add_model == 'LambdaRank':
        results['ndcg_LambdaRank'] = [np.mean(ndcg_lrank_l), np.std(ndcg_lrank_l)]

    with open(results_path + '/results_label_{}.json'.format(n_train), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get some run values.')
    parser.add_argument('--test', type=int, default=0, help='Run in test mode')
    parser.add_argument('--path', type=str, default="mslr_data", help='Path to data')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not os.path.exists("output"):
        os.makedirs("output")

    test = False
    if args.test == 1:
        test = True

    output_path = 'mslr_data'
    results_path = output_path + "/results"
    data_path = args.path

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_path + "/data"):
        os.makedirs(output_path + "/data")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    train_list = np.logspace(2, 6, num=10)
    train_list = np.concatenate([np.logspace(1, 2, num=3)[1:2], train_list])
    for n_train in train_list:
        test_rankers(n_train=int(n_train), overwrite=True, test=test)
        if test: break
