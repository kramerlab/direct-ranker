import os
from Rankers.models.RankNet import RankNet
from Rankers.helpers import nDCG_cls, readData, colors_points, colors_bar

import numpy as np
import pandas as pd
import tensorflow as tf

import argparse


def plot(args):
    sig_adam_ranknet = np.load(f"{args.out_path}/sigmoid_MSLR10K_Adam_list_of_ndcg.npy")
    hsig_adam_ranknet = np.load(f"{args.out_path}/hard_sigmoid_MSLR10K_Adam_list_of_ndcg.npy")
    tanh_adam_ranknet = np.load(f"{args.out_path}/tanh_MSLR10K_Adam_list_of_ndcg.npy")
    linear_adam_ranknet = np.load(f"{args.out_path}/linear_MSLR10K_Adam_list_of_ndcg.npy")
    sig_sgd_ranknet = np.load(f"{args.out_path}/sigmoid_MSLR10K_SGD_list_of_ndcg.npy")
    hsig_sgd_ranknet = np.load(f"{args.out_path}/hard_sigmoid_MSLR10K_SGD_list_of_ndcg.npy")
    tanh_sgd_ranknet = np.load(f"{args.out_path}/tanh_MSLR10K_SGD_list_of_ndcg.npy")
    linear_sgd_ranknet = np.load(f"{args.out_path}/linear_MSLR10K_SGD_list_of_ndcg.npy")

    d = {
        'Optimizer / Activation': ["Gradient descent", "Adam"],
        'Sigmoid': [f"{round(np.mean(sig_sgd_ranknet),3)}({int(round(np.std(sig_sgd_ranknet),3)*1000)})",
                    f"{round(np.mean(sig_adam_ranknet),3)}({int(round(np.std(sig_adam_ranknet),3)*1000)})"],
        'Hard_Sigmoid': [f"{round(np.mean(hsig_sgd_ranknet),3)}({int(round(np.std(hsig_sgd_ranknet),3)*1000)})",
                    f"{round(np.mean(hsig_adam_ranknet),3)}({int(round(np.std(hsig_adam_ranknet),3)*1000)})"],
        'Tanh':    [f"{round(np.mean(tanh_sgd_ranknet),3)}({int(round(np.std(tanh_sgd_ranknet),3)*1000)})",
                    f"{round(np.mean(tanh_adam_ranknet),3)}({int(round(np.std(tanh_adam_ranknet),3)*1000)})"],
        'Linear':    [f"{round(np.mean(linear_sgd_ranknet),3)}({int(round(np.std(linear_sgd_ranknet),3)*1000)})",
                    f"{round(np.mean(linear_adam_ranknet),3)}({int(round(np.std(linear_adam_ranknet),3)*1000)})"],
        }

    print(pd.DataFrame(data=d).to_latex(index=False))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get some run values.')
    parser.add_argument('--activation', type=str, default='sigmoid', help='Get ranking activation')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Get optimizer name')
    parser.add_argument('--epoch', type=int, default=100, help='Get number of epoch')
    parser.add_argument('--data', type=str, default='MSLR10K', help='Get data name')
    parser.add_argument('--path', type=str, default='', help='Get path to data')
    parser.add_argument('--out_path', type=str, default='ranknet_eval', help='Get path to data')
    parser.add_argument('--cpu', type=int, default=1, help='Use CPU or GPU')
    parser.add_argument('--plot', type=int, default=0, help='Plot the resutls')

    args = parser.parse_args()

    if args.plot == 1:
        plot(args)
        exit()

    if args.cpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        os.makedirs(args.out_path + "/data")

    if args.optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD
    if args.optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam

    list_of_train = [f"{args.path}/Fold1/train.txt", f"{args.path}/Fold2/train.txt",
                    f"{args.path}/Fold3/train.txt", f"{args.path}/Fold4/train.txt",
                    f"{args.path}/Fold5/train.txt"]
    list_of_test = [f"{args.path}/Fold1/test.txt", f"{args.path}/Fold2/test.txt",
                    f"{args.path}/Fold3/test.txt", f"{args.path}/Fold4/test.txt",
                    f"{args.path}/Fold5/test.txt"]

    parameters = {
        'hidden_layers_dr': [32, 20, 5],
        'epoch': args.epoch,
        'feature_activation_dr': 'sigmoid',
        'verbose': 2,
        'optimizer': optimizer,
        'ranking_activation_dr': args.activation,
    }

    list_of_ndcg = []
    for foldidx in range(len(list_of_train)):
        print("FOLD ", foldidx)

        if args.data == "MSLR10K" or args.data == "test":
            number_features = 136

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
                number_features=number_features
            )
            x_test, y_test, q_test = readData(
                path=list_of_test[foldidx],
                binary=False,
                at=10,
                number_features=number_features
            )

            np.save(f"{args.out_path}/data/{args.data}_x_train_fold{foldidx+1}", x_train)
            np.save(f"{args.out_path}/data/{args.data}_y_train_fold{foldidx+1}", y_train)

            np.save(f"{args.out_path}/data/{args.data}_x_test_fold{foldidx+1}", x_test)
            np.save(f"{args.out_path}/data/{args.data}_y_test_fold{foldidx+1}", y_test)
            np.save(f"{args.out_path}/data/{args.data}_q_test_fold{foldidx+1}", q_test)

        print("Training")
        ranker = RankNet(**parameters, num_features=x_train.shape[1])

        ranker.fit(x_train, y_train)

        print("Eval")
        nDCG_l = []
        pred = ranker.predict_proba(x_test)
        for n in np.unique(q_test):
            nDCG_l.append(nDCG_cls(pred[q_test==n], x_test[q_test==n], y_test[q_test==n], at=10, prediction=True))

        print(f"Test on Fold {foldidx}: NDCG@10 = {np.mean(nDCG_l)}")

        list_of_ndcg.append(np.mean(nDCG_l))

    np.save(f"{args.out_path}/{args.activation}_{args.data}_{args.optimizer}_list_of_ndcg", list_of_ndcg)
