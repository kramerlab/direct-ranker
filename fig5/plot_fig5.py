import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json


def plot(path = 'synth_data', label=False):
    results_path = path + "/results"
    n_test = 10000
    folds = [0, 1, 2, 3, 4]
    models = ["ListNet", "DirectRanker", "RankNet", "LambdaRank"]
    msize = 5
    colors_points = [
        (57 / 255, 106 / 255, 177 / 255),    # blue
        (218 / 255, 124 / 255, 48 / 255),    # orange
        (62 / 255, 150 / 255, 81 / 255),     # green
        (204 / 255, 37 / 255, 41 / 255),     # red
        (83 / 255, 81 / 255, 84 / 255),      # black
        (107 / 255, 76 / 255, 154 / 255),    # purple
        (146 / 255, 36 / 255, 40 / 255),     # wine
        (148 / 255, 139 / 255, 61 / 255)     # gold
    ]
    colors_bar = [
        (114 / 255, 147 / 255, 203 / 255),    # blue
        (225 / 255, 151 / 255, 76 / 255),     # orange
        (132 / 255, 186 / 255, 91 / 255),     # green
        (211 / 255, 94 / 255, 96 / 255),      # red
        (128 / 255, 133 / 255, 133 / 255),    # black
        (144 / 255, 103 / 255, 167 / 255),    # purple
        (171 / 255, 104 / 255, 87 / 255),     # wine
        (204 / 255, 194 / 255, 16 / 255)      # gold
    ]

    train_list = np.logspace(2, 6, num=10)
    if label:
        train_list = np.concatenate([np.logspace(1, 2, num=3)[:2], train_list])

    for n_train in train_list:
        if path == 'synth_data' and label == False:
            cur_path = results_path + "/results_{}_{}.json".format(int(n_train), n_test)
        elif label and path == 'synth_data':
            cur_path = results_path + "/results_label_{}_{}.json".format(int(n_train), n_test)
        elif label:
            cur_path = results_path + "/results_label_{}.json".format(int(n_train))
        else:
            cur_path = results_path + "/results_{}.json".format(int(n_train))

        if not os.path.isfile(cur_path):
            continue
        if label: 
            if n_train > 50000: continue
            if path == 'mslr_data':
                n_train = n_train/112871
            else:
                n_train = n_train/50000

        with open(cur_path) as json_file: 
            data = json.load(json_file)
            for model in models:
                try:
                    nDCG = data["ndcg_{}".format(model)]
                except:
                    print("No model {}".format(model)) 
                    continue
                if model == "ListNet":
                    plt.scatter(n_train, nDCG[0], c=colors_points[0], marker='s', label=model, s=msize)
                if model == "DirectRanker":
                    plt.scatter(n_train, nDCG[0], c=colors_points[1], marker='x', label=r"RankNet$^*$", s=msize)
                if model == "RankNet":
                    plt.scatter(n_train, nDCG[0], c=colors_points[2], marker='>', label=model, s=msize)
                if model == "LambdaRank":
                    plt.scatter(n_train, nDCG[0], c=colors_points[3], marker='>', label=model, s=msize)
                plt.errorbar(n_train, nDCG[0], yerr=nDCG[1], c=colors_bar[4], alpha=0.5, linestyle="None", zorder=0, label='_nolegend_')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper left")
    plt.xscale("log")
    if label:
        plt.xlabel("ratio \# relevant and \# unrelevant", fontsize=7)
    else:
        plt.xlabel("\# instances in training", fontsize=7)
    if path == 'synth_data':
        plt.ylabel("nDCG@20", fontsize=7)
    else:
        plt.ylabel("nDCG@10", fontsize=7)
    if label:
        plt.savefig('training_label_{}.pdf'.format(path))
        plt.savefig('training_label_{}.png'.format(path))
    else:
        plt.savefig('training_size_{}.pdf'.format(path))
        plt.savefig('training_size_{}.png'.format(path))
    plt.close()


plot('synth_data')
plot('synth_data', True)
plot('mslr_data')
plot('mslr_data', True)
