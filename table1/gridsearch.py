import sys
from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

import os
from Rankers.models.RankNet import RankNet
from Rankers.models.DirectRanker import DirectRanker
from Rankers.models.ListNet import ListNet
from Rankers.models.LambdaRank import LambdaRank
from Rankers.models.LambdaMart2 import LambdaMart
from Rankers.models.AdaRank import AdaRank
from Rankers.helpers import nDCG_cls, readData, AvgP_cls, readDataV1, nDCGScorer_cls, MAP_cls

import numpy as np
import tensorflow as tf
from functools import partial

from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GridSearchCV, train_test_split

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get some run values.')
    parser.add_argument('--data', type=str, default='MSLR10K', help='Get data name')
    parser.add_argument('--model', type=str, default='DirectRanker', help='Get model name')
    parser.add_argument('--path', type=str, default='', help='Get path to data')
    parser.add_argument('--out_path', type=str, default='gridsearch', help='Get path to data')
    parser.add_argument('--cpu', type=int, default=1, help='Use CPU or GPU')
    parser.add_argument('--jobs', type=int, default=1, help='Use CPUs for running')
    parser.add_argument('--ttest', type=int, default=0, help='Using ttest')
    parser.add_argument('--runNum', type=int, default=0, help='Run Number')
    parser.add_argument('--binary', type=int, default=1, help='binary')
    parser.add_argument('--startFold', type=int, default=-1, help='skip the other folds')

    args = parser.parse_args()

    if args.cpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if args.binary == 1:
        binary = True
    else:
        binary = False

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        os.makedirs(args.out_path + "/data")

    list_of_train = [f"{args.path}/Fold1/train.txt", f"{args.path}/Fold2/train.txt",
                    f"{args.path}/Fold3/train.txt", f"{args.path}/Fold4/train.txt",
                    f"{args.path}/Fold5/train.txt"]
    list_of_test = [f"{args.path}/Fold1/test.txt", f"{args.path}/Fold2/test.txt",
                    f"{args.path}/Fold3/test.txt", f"{args.path}/Fold4/test.txt",
                    f"{args.path}/Fold5/test.txt"]
    list_of_vali = [f"{args.path}/Fold1/vali.txt", f"{args.path}/Fold2/vali.txt",
                    f"{args.path}/Fold3/vali.txt", f"{args.path}/Fold4/vali.txt",
                    f"{args.path}/Fold5/vali.txt"]

    if args.ttest == 1:
        ttest = True
        numOutFolds = 15
        numInFolds = 3
    else:
        ttest = False
        numOutFolds = 5
        numInFolds = 5

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
    for foldidx in range(numOutFolds):
        print("FOLD ", foldidx)

        if args.startFold != -1 and args.startFold > foldidx:
            print("We skip")
            fold_dict = np.load(f"{args.out_path}/cv_results_{args.data}_{args.model}_fold{foldidx}_ttest{ttest}_binary{binary}_runNum{args.runNum}.npy", allow_pickle=True)[()]
            list_of_ndcg.append(fold_dict["ndcg"])
            list_of_map.append(fold_dict["map"])
            print(f"Test on Fold {foldidx}: NDCG@10 = {fold_dict['ndcg']} MAP = {fold_dict['map']}")
            continue

        if args.data == "MSLR10K":
            number_features = 136
            cut = 1.5
        if args.data == "test":
            number_features = 136
            cut = 0.5
        if args.data == "MQ2007" or args.data == "MQ2008":
            number_features = 46
            cut = 0.5

        if foldidx > 4:
            readFoldidx = 0
        else:
            readFoldidx = foldidx

        # read data
        print("Reading data")
        if os.path.isfile(f"{args.out_path}/data/{args.data}_q_vali_fold{readFoldidx+1}_binary{binary}.npy"):
            x_train = np.load(f"{args.out_path}/data/{args.data}_x_train_fold{readFoldidx+1}_binary{binary}.npy")
            y_train = np.load(f"{args.out_path}/data/{args.data}_y_train_fold{readFoldidx+1}_binary{binary}.npy")
            q_train = np.load(f"{args.out_path}/data/{args.data}_q_train_fold{readFoldidx+1}_binary{binary}.npy")

            x_test = np.load(f"{args.out_path}/data/{args.data}_x_test_fold{readFoldidx+1}_binary{binary}.npy")
            y_test = np.load(f"{args.out_path}/data/{args.data}_y_test_fold{readFoldidx+1}_binary{binary}.npy")
            q_test = np.load(f"{args.out_path}/data/{args.data}_q_test_fold{readFoldidx+1}_binary{binary}.npy")

            x_vali = np.load(f"{args.out_path}/data/{args.data}_x_vali_fold{readFoldidx+1}_binary{binary}.npy")
            y_vali = np.load(f"{args.out_path}/data/{args.data}_y_vali_fold{readFoldidx+1}_binary{binary}.npy")
            q_vali = np.load(f"{args.out_path}/data/{args.data}_q_vali_fold{readFoldidx+1}_binary{binary}.npy")
        else:
            x_train, y_train, q_train = readData(
                path=list_of_train[readFoldidx],
                binary=binary,
                at=10,
                number_features=number_features,
                cut=cut
            )

            x_test, y_test, q_test = readData(
                path=list_of_test[readFoldidx],
                binary=binary,
                at=10,
                number_features=number_features,
                cut=cut
            )

            x_vali, y_vali, q_vali = readData(
                path=list_of_vali[readFoldidx],
                binary=binary,
                at=10,
                number_features=number_features,
                cut=cut
            )

            np.save(f"{args.out_path}/data/{args.data}_x_train_fold{readFoldidx+1}_binary{binary}", x_train)
            np.save(f"{args.out_path}/data/{args.data}_y_train_fold{readFoldidx+1}_binary{binary}", y_train)
            np.save(f"{args.out_path}/data/{args.data}_q_train_fold{readFoldidx+1}_binary{binary}", q_train)

            np.save(f"{args.out_path}/data/{args.data}_x_test_fold{readFoldidx+1}_binary{binary}", x_test)
            np.save(f"{args.out_path}/data/{args.data}_y_test_fold{readFoldidx+1}_binary{binary}", y_test)
            np.save(f"{args.out_path}/data/{args.data}_q_test_fold{readFoldidx+1}_binary{binary}", q_test)

            np.save(f"{args.out_path}/data/{args.data}_x_vali_fold{readFoldidx+1}_binary{binary}", x_vali)
            np.save(f"{args.out_path}/data/{args.data}_y_vali_fold{readFoldidx+1}_binary{binary}", y_vali)
            np.save(f"{args.out_path}/data/{args.data}_q_vali_fold{readFoldidx+1}_binary{binary}", q_vali)

        if ttest:
            x = np.concatenate([x_train, x_test, x_vali])
            y = np.concatenate([y_train, y_test, y_vali])
            q = np.concatenate([q_train, q_test, q_vali])
            x_train, x_test, y_train, y_test, q_train, q_test = train_test_split(x, y, q, test_size=0.33, shuffle=True)

        print("Preprocessing")
        scaler = QuantileTransformer(output_distribution="normal")
        x_train = scaler.fit_transform(x_train) / 3
        x_test = scaler.transform(x_test) / 3
        binary_y_test = []
        # if we have multi class in training we do binary for eval
        if not binary:
            for yi in y_test:
                if int(yi) > cut:
                    binary_y_test.append(1)
                else:
                    binary_y_test.append(0)
            y_test = np.array(binary_y_test)

        print("Training")
        if args.model == "RankNet":
            parameters['optimizer'] = [tf.keras.optimizers.SGD]
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
            x_train, y_train, q_train = readDataV1(
                data_path=list_of_train[readFoldidx],
                binary=True,
                at=10,
                number_features=number_features,
                bin_cutoff=cut,
                synth_data=True,
                cut_zeros=True
            )
            x_test, y_test, q_test = readDataV1(
                data_path=list_of_test[readFoldidx],
                binary=True,
                at=10,
                number_features=number_features,
                bin_cutoff=cut,
                synth_data=True,
                cut_zeros=True
            )

            if ttest:
                x = np.concatenate([x_train, x_test, x_vali])
                y = np.concatenate([y_train, y_test, y_vali])
                q = np.concatenate([q_train, q_test, q_vali])
                x_train, x_test, y_train, y_test, q_train, q_test = train_test_split(x, y, q, test_size=0.33, shuffle=True)

            parameters_test = {"max_steps": [100], "print_step": [10], "start_batch_size": [10], "end_batch_size": [100]}
            ranker = DirectRankerV1()

        if args.model == "ListNet":
            parameters['optimizer'] = [tf.keras.optimizers.SGD]
            ranker = ListNet(num_features=x_train.shape[1])

        if args.model == "LambdaRank":
            ranker = LambdaRank(num_features=x_train.shape[1])

        if args.model == "LambdaMart":
            parameters_test = {
                "number_of_trees": [5],
                "learning_rate": [0.1]
            }
            parameters = {
                "number_of_trees": [5, 10],
                "max_depth": [4],
                "learning_rate": [0.1, 0.01],
            }
            ranker = LambdaMart()
            x_train = np.concatenate([q_train.reshape(len(q_train), 1), x_train], axis=1)
            x_test = np.concatenate([q_test.reshape(len(q_test), 1), x_test], axis=1)

        if args.model == "AdaRank":
            parameters_test = {
                "T": [100],
                "verbose": [True],
                "estop": [10]
            }
            parameters = {
                "T": [100],
                "verbose": [True],
                "estop": [0, 10, 20]
            }
            ranker = AdaRank()
            x_train = np.concatenate([q_train.reshape(len(q_train), 1), x_train], axis=1)
            x_test = np.concatenate([q_test.reshape(len(q_test), 1), x_test], axis=1)

        if args.data == "test":
            parameters = parameters_test

        if args.data == "test":
            ranker = GridSearchCV(ranker, parameters, cv=2, n_jobs=args.jobs, verbose=1, scoring=scoring, refit='NDGC@10', return_train_score=False)
        else:
            ranker = GridSearchCV(ranker, parameters, cv=numInFolds, n_jobs=args.jobs, verbose=1, scoring=scoring, refit='NDGC@10', return_train_score=False)

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
                if ndcg == 0 and (args.data == "MQ2008" or args.data == "MQ2007" or args.data == "MSLR10K"):
                    ndcg == 1
                if mapv == 0 and (args.data == "MQ2008" or args.data == "MQ2007" or args.data == "MSLR10K"):
                    mapv == 1
                nDCG_l.append(ndcg)
                map_l.append(mapv)

        ranker.cv_results_["ndcg"] = np.mean(nDCG_l)
        ranker.cv_results_["map"] = np.mean(map_l)

        print(f"Test on Fold {foldidx}: NDCG@10 = {np.mean(nDCG_l)} MAP = {np.mean(map_l)}")
        #print(ranker.cv_results_)

        list_of_ndcg.append(np.mean(nDCG_l))
        list_of_map.append(np.mean(map_l))

        np.save(f"{args.out_path}/cv_results_{args.data}_{args.model}_fold{foldidx}_ttest{ttest}_binary{binary}_runNum{args.runNum}", ranker.cv_results_)

    np.save(f"{args.out_path}/{args.data}_{args.model}_list_of_ndcg_ttest{ttest}_binary{binary}_runNum{args.runNum}", list_of_ndcg)
    np.save(f"{args.out_path}/{args.data}_{args.model}_list_of_map_ttest{ttest}_binary{binary}_runNum{args.runNum}", list_of_map)
