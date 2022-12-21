import os
from Rankers.models.RankNet import RankNet
from Rankers.models.DirectRanker import DirectRanker
from Rankers.models.ListNet import ListNet
from Rankers.models.LambdaRank import LambdaRank
from Rankers.helpers import nDCG_cls, readData, AvgP_cls, readDataV1, nDCGScorer_cls, MAP_cls

import numpy as np
import tensorflow as tf
from functools import partial

from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GridSearchCV

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get some run values.')
    parser.add_argument('--data', type=str, default='MSLR10K', help='Get data name')
    parser.add_argument('--model', type=str, default='DirectRanker', help='Get model name')
    parser.add_argument('--path', type=str, default='', help='Get path to data')
    parser.add_argument('--out_path', type=str, default='gridsearch', help='Get path to data')
    parser.add_argument('--cpu', type=int, default=1, help='Use CPU or GPU')
    parser.add_argument('--jobs', type=int, default=1, help='Use CPUs for running')

    args = parser.parse_args()

    if args.cpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        os.makedirs(args.out_path + "/data")

    list_of_train = [f"{args.path}/Fold1/train.txt", f"{args.path}/Fold2/train.txt",
                    f"{args.path}/Fold3/train.txt", f"{args.path}/Fold4/train.txt",
                    f"{args.path}/Fold5/train.txt"]
    list_of_test = [f"{args.path}/Fold1/test.txt", f"{args.path}/Fold2/test.txt",
                    f"{args.path}/Fold3/test.txt", f"{args.path}/Fold4/test.txt",
                    f"{args.path}/Fold5/test.txt"]

    parameters = {
        'hidden_layers_dr': [[10, 5], [20, 5], [30, 10, 5], [40, 30, 10, 5], [200, 50, 30, 10, 5]],
        'feature_activation_dr': ["tanh", "sigmoid", "linear"],
        'epoch': [5, 10, 30],
        'ranking_activation_dr': ['sigmoid'],
        'verbose': [2],
        'optimizer': [tf.keras.optimizers.Adam],
        'learning_rate': [0.001, 0.0001]
    }

    parameters_test = {'epoch': [1]}

    if args.model == "DirectRankerV1":
        nDCGScorer10 = partial(nDCGScorer_cls, at=10)
        scoring = {'NDGC@10': nDCGScorer10, 'MAP': MAP_cls}
    else:
        scoring = {'NDGC@10': nDCG_cls}

    list_of_ndcg = []
    list_of_map = []
    for foldidx in range(len(list_of_train)):
        print("FOLD ", foldidx)

        if args.data == "MSLR10K":
            number_features = 136
            cut = 1.5
        if args.data == "test":
            number_features = 136
            cut = 0.5
        if args.data == "MQ2007" or args.data == "MQ2008":
            number_features = 46
            cut = 0.5
            
        # read data
        print("Reading data")
        if os.path.isfile(f"{args.out_path}/data/{args.data}_x_train_fold{foldidx+1}.npy"):
            x_train = np.load(f"{args.out_path}/data/{args.data}_x_train_fold{foldidx+1}.npy")
            y_train = np.load(f"{args.out_path}/data/{args.data}_y_train_fold{foldidx+1}.npy")

            x_test = np.load(f"{args.out_path}/data/{args.data}_x_test_fold{foldidx+1}.npy")
            y_test = np.load(f"{args.out_path}/data/{args.data}_y_test_fold{foldidx+1}.npy")
            q_test = np.load(f"{args.out_path}/data/{args.data}_q_test_fold{foldidx+1}.npy")
        else:
            x_train, y_train, _ = readData(
                path=list_of_train[foldidx],
                binary=True,
                at=10,
                number_features=number_features,
                cut=cut
            )

            x_test, y_test, q_test = readData(
                path=list_of_test[foldidx],
                binary=True,
                at=10,
                number_features=number_features,
                cut=cut
            )

            np.save(f"{args.out_path}/data/{args.data}_x_train_fold{foldidx+1}", x_train)
            np.save(f"{args.out_path}/data/{args.data}_y_train_fold{foldidx+1}", y_train)

            np.save(f"{args.out_path}/data/{args.data}_x_test_fold{foldidx+1}", x_test)
            np.save(f"{args.out_path}/data/{args.data}_y_test_fold{foldidx+1}", y_test)
            np.save(f"{args.out_path}/data/{args.data}_q_test_fold{foldidx+1}", q_test)

        print("Preprocessing")
        scaler = QuantileTransformer(output_distribution="normal")
        x_train = scaler.fit_transform(x_train) / 3
        x_test = scaler.transform(x_test) / 3

        print("Training")
        if args.model == "RankNet":
            ranker = RankNet(num_features=x_train.shape[1])

        if args.model == "DirectRanker":
            parameters['ranking_activation_dr'] = ['tanh']
            parameters['drop_out'] = [0., 0.2, 0.5]
            parameters['kernel_regularizer_dr'] =  [0., 0.001, 0.01]
            ranker = DirectRanker(num_features=x_train.shape[1])
        
        if args.model == "DirectRankerV1":
            from Rankers.models.DirectRankerV1 import DirectRankerV1
            parameters = {
                'hidden_layers': [[10], [20], [30], [50], [100], [50,5], [100, 5], [100,50,5], [50,25,5], [100,50,25,5], [200,100,50,25,5], [100,100,50,25,5]],
                'weight_regularization': [0., 0.0001, 0.001, 0.01, 0.1, 0.999],
                'early_stopping': [False, True],
                'dropout': [0., 0.5]
            }
            x_train, y_train, _ = readDataV1(
                data_path=list_of_train[foldidx],
                binary=True,
                at=10,
                number_features=number_features,
                bin_cutoff=cut,
                synth_data=True,
                cut_zeros=True
            )
            x_test, y_test, _ = readDataV1(
                data_path=list_of_test[foldidx],
                binary=True,
                at=10,
                number_features=number_features,
                bin_cutoff=cut,
                synth_data=True,
                cut_zeros=True
            )
            parameters_test = {"max_steps": [100], "print_step": [10], "start_batch_size": [10], "end_batch_size": [100]}
            ranker = DirectRankerV1()

        if args.model == "ListNet":
            ranker = ListNet(num_features=x_train.shape[1])

        if args.model == "LambdaRank":
            ranker = LambdaRank(num_features=x_train.shape[1])

        if args.data == "test":
            parameters = parameters_test

        if args.data == "test":
            ranker = GridSearchCV(ranker, parameters, cv=2, n_jobs=args.jobs, verbose=1, scoring=scoring, refit='NDGC@10', return_train_score=False)
        else:
            ranker = GridSearchCV(ranker, parameters, cv=5, n_jobs=args.jobs, verbose=1, scoring=scoring, refit='NDGC@10', return_train_score=False)
        
        if args.model == "DirectRankerV1":
            ranker.fit(x_train, y_train, ranking=False)
        else:
            ranker.fit(x_train, y_train)

        print("Eval")
        if args.model == "DirectRankerV1":
            map_l = MAP_cls(ranker.best_estimator_, x_test, y_test)
            nDCG_l = nDCGScorer10(ranker.best_estimator_, x_test, y_test)
        else:
            nDCG_l = []
            map_l = []
            pred = ranker.best_estimator_.predict_proba(x_test)
            for n in np.unique(q_test):
                ndcg = nDCG_cls(pred[q_test==n], x_test[q_test==n], y_test[q_test==n], at=10, prediction=True)
                mapv = AvgP_cls(pred[q_test==n], x_test[q_test==n], y_test[q_test==n], prediction=True)
                # exclude if there was a querry with only one class
                if ndcg == 0 and (args.data == "MQ2008" or args.data == "MQ2007"):
                    ndcg == 1
                if mapv == 0 and (args.data == "MQ2008" or args.data == "MQ2007"):
                    mapv == 1
                nDCG_l.append(ndcg)
                map_l.append(mapv)

        print(f"Test on Fold {foldidx}: NDCG@10 = {np.mean(nDCG_l)} MAP = {np.mean(map_l)}")
        print(ranker.cv_results_)

        list_of_ndcg.append(np.mean(nDCG_l))
        list_of_map.append(np.mean(map_l))

        np.save(f"{args.out_path}/cv_results_{args.data}_{args.model}_fold{foldidx}", ranker.cv_results_)

    np.save(f"{args.out_path}/{args.data}_{args.model}_list_of_ndcg", list_of_ndcg)
    np.save(f"{args.out_path}/{args.data}_{args.model}_list_of_map", list_of_map)
