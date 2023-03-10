import os
import time
import argparse

import numpy as np

from Rankers.helpers import readData, readDataV1
from Rankers.models.DirectRanker import DirectRanker
from Rankers.models.RankNet import RankNet


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get some run values.')
    parser.add_argument('--model', type=str, default='DirectRanker', help='Get model name')
    parser.add_argument('--cpu', type=int, default=1, help='Use CPU or GPU')
    parser.add_argument('--path', type=str, default="", help='Use CPU or GPU')

    args = parser.parse_args()

    if args.cpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not os.path.exists("output"):
        os.makedirs("output")
        os.makedirs("output/data")

    list_of_train = [f"{args.path}/Fold1/train.txt", f"{args.path}/Fold2/train.txt",
                    f"{args.path}/Fold3/train.txt", f"{args.path}/Fold4/train.txt",
                    f"{args.path}/Fold5/train.txt"]

    for idx, train in enumerate(list_of_train):

        # Load model
        print("Load Model")
        if args.model == "DirectRankerV1":
            from Rankers.models.DirectRankerV1 import DirectRankerV1
            ranker = DirectRankerV1(
                # For debugging
                # max_steps=100,
                print_step=0,
                start_batch_size=3,
                end_batch_size=5,
                start_qids=10,
                end_qids=100,
                weight_regularization=0.0001,
                feature_bias=True,
                hidden_layers=[50]
            )

        if args.model == "DirectRanker":
            ranker = DirectRanker(
                epoch=10,
                num_features=136
            )

        if args.model == "RankNet":
            ranker = RankNet(
                epoch=10,
                num_features=136
            )

        print("Reading data")
        start = time.time()
        if os.path.isfile(f"output/data/MSLR-WEB10K_x_train_fold{idx+1}.npy"):
            x_train = np.load(f"output/data/MSLR-WEB10K_x_train_fold{idx+1}.npy")
            y_train = np.load(f"output/data/MSLR-WEB10K_y_train_fold{idx+1}.npy")
        else:
            if args.model == "DirectRankerV1":
                x_train, y_train, _ = readDataV1(
                    data_path=train,
                    binary=True,
                    at=10,
                    number_features=136,
                    synth_data=True,
                    cut_zeros=True
                )
            else:
                x_train, y_train, _ = readData(
                    path=train,
                    binary=True,
                    at=10,
                    number_features=136
                )

            np.save(f"output/data/MSLR-WEB10K_x_train_fold{idx+1}", x_train)
            np.save(f"output/data/MSLR-WEB10K_y_train_fold{idx+1}", y_train)

        if args.model == "DirectRankerV1":
            ranker.fit(x_train, y_train, ranking=False)
        else:
            ranker.fit(x_train, y_train)
        end = time.time()

        print(end - start)

        with open(f"output/{args.model}_time_fold{idx+1}.txt", "w") as f:
            f.write(str(end-start))
