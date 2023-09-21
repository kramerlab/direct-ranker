import os
import tensorflow as tf

from Rankers.models.DirectRanker import DirectRanker
from Rankers.models.RankNet import RankNet
from Rankers.models.ListNet import ListNet
from Rankers.helpers import nDCG_cls, readData
from Rankers.models.LambdaRank import LambdaRank

import numpy as np
from sklearn.preprocessing import QuantileTransformer
from time import time

import json
import argparse


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
        if os.path.isfile(results_path + "/results_{}.json".format(n_train)):
            if not (add_model == None):
                results = json.load(open(results_path + "/results_{}.json".format(n_train)))
            elif overwrite:
                os.remove(results_path + "/results_{}.json".format(n_train)) 
            else:
                print("skip training for n_relevance {}".format(n_train))
                return 0

        N = 50
        x = []
        y = []
        x_test = []
        y_test = []

        # read data
        print("Reading data")
        start = time()
        if os.path.isfile(output_path + "/data/x_train_fold{}.npy".format(fold+1)):
            x_train = np.load(output_path + "/data/x_train_fold{}.npy".format(fold+1))
            y_train = np.load(output_path + "/data/y_train_fold{}.npy".format(fold+1))
            
            x_test = np.load(output_path + "/data/x_test_fold{}.npy".format(fold+1))
            y_test = np.load(output_path + "/data/y_test_fold{}.npy".format(fold+1))
            q_test = np.load(output_path + "/data/q_test_fold{}.npy".format(fold+1))
        else:
            x_train, y_train, _ = readData(path=data_path + "/Fold{}/train.txt".format(fold+1))
            x_test, y_test, q_test = readData(path=data_path + "/Fold{}/test.txt".format(fold+1), binary=False)
        
            np.save(output_path + "/data/x_train_fold{}".format(fold+1), x_train)
            np.save(output_path + "/data/y_train_fold{}".format(fold+1), y_train)
            
            np.save(output_path + "/data/x_test_fold{}".format(fold+1), x_test)
            np.save(output_path + "/data/y_test_fold{}".format(fold+1), y_test)
            np.save(output_path + "/data/q_test_fold{}".format(fold+1), q_test)
        end = time()
        mins = int((end - start) / 60)
        secs = end - start - 60 * mins
        print("Finished in " + str(mins) + " min " + str(secs) + " s")

        print("Preparing data n_train {} on fold {}".format(n_train, fold))
        if len(x_train) < n_train: 
            n_train = len(x_train)
            x = x_train
            y = y_train
        else:
            idx = np.random.randint(0, len(x_train), n_train)
            x = x_train[idx]
            y = y_train[idx]
        
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
            dr.fit(x, y)
        if not ('ndcg_RankNet' in results) or add_model == 'RankNet':
            ranknet.fit(x, y)
        if not ('ndcg_ListNet' in results) or add_model == 'ListNet':
            lnet.fit(x, tf.cast(y, tf.float32))
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

    with open(results_path + '/results_{}.json'.format(n_train), 'w', encoding='utf-8') as f:
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

    for n_train in np.logspace(2, 6, num=10):
        test_rankers(n_train=int(n_train), overwrite=False, test=test)
        if test: break
