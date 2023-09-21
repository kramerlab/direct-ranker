import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import os


def get_ranklib(ttest=False):
    # get ranklib results
    for algo in ["LambdaMart", "AdaRank", "ListNet", "RankNet"]:
        for pathname in ["MQ2007", "MQ2008", "MSLR10K"]:
            if ttest:
                folds = [f"Fold{i}" for i in range(1,16)]
                best_map = [0 for i in range(1,16)]
                best_map_para = [0 for i in range(1,16)]
                best_ndcg = [0 for i in range(1,16)]
                best_ndcg_para = [0 for i in range(1,16)]
            if not ttest:
                folds = [f"Fold{i}" for i in range(1,6)]
                best_map = [0 for i in range(1,6)]
                best_map_para = [0 for i in range(1,6)]
                best_ndcg = [0 for i in range(1,6)]
                best_ndcg_para = [0 for i in range(1,6)]
            for idx_fold, fold in enumerate(folds):
                if not os.path.exists("results_ranklib/" + pathname + "/"): continue
                for filename in os.listdir("results_ranklib/" + pathname + "/"):
                    if algo not in filename:
                        continue
                    if fold not in filename:
                        continue
                    with open("results_ranklib/" + pathname + "/" + filename, "r") as file:
                        for line in file:
                            if "Avg." in line:
                                #print(line, (line.replace("\n","").split()[2]))
                                if "MAP" in filename:
                                    if best_map[idx_fold] < float(line.replace("\n","").split()[2]):
                                        best_map[idx_fold] = float(line.replace("\n","").split()[2])
                                        best_map_para[idx_fold] = filename
                                else:
                                    if best_ndcg[idx_fold] < float(line.replace("\n","").split()[2]):
                                        best_ndcg[idx_fold] = float(line.replace("\n","").split()[2])
                                        best_ndcg_para[idx_fold] = filename
            print(algo, pathname)
            print("MAP " + str(best_map), f"{round(np.mean(best_map),3)}({int(round(np.std(best_map),3)*1000)})")
            print("NDCG@10  " + str(best_ndcg), f"{round(np.mean(best_ndcg),3)}({int(round(np.std(best_ndcg),3)*1000)})")

#get_ranklib(ttest=True)

# get tensorflow V2 results
results = {}
for model in ["DirectRanker", "ListNet", "RankNet", "DirectRankerV1"]:
    cur_dict = {}
    for data in ["MSLR10K", "MQ2008", "MQ2007"]:
        print(model, data, "NDCG", np.load(f"gridsearch/{data}_{model}_list_of_ndcg.npy", allow_pickle=True))
        print(model, data, "MAP", np.load(f"gridsearch/{data}_{model}_list_of_map.npy", allow_pickle=True))
        cur_dict[data] = {
            "NDCG-mean": np.mean(np.load(f"gridsearch/{data}_{model}_list_of_ndcg.npy", allow_pickle=True)),
            "NDCG-std": np.std(np.load(f"gridsearch/{data}_{model}_list_of_ndcg.npy", allow_pickle=True)),
            "MAP-mean": np.mean(np.load(f"gridsearch/{data}_{model}_list_of_map.npy", allow_pickle=True)),
            "MAP-std": np.std(np.load(f"gridsearch/{data}_{model}_list_of_map.npy", allow_pickle=True))
        }
    results[model] = cur_dict

print(cur_dict)

# get ttest results
def bengio_nadeau_test(model1, model2, alpha=0.05):
    n = len(model1)
    assert len(model1) == len(model2)
    differences = [model1[i] - model2[i] for i in range(n)]
    divisor = 1 / n * np.sum(differences)
    # critical to change this if values change in run_experiment
    test_train_ratio = 0.33
    denominator = np.sqrt(1 / n + test_train_ratio) * np.std(differences)
    t_stat = divisor / denominator
    # degrees of freedom
    df = n - 1
    # critical value
    cv = ss.t.ppf(1.0 - alpha, df)
    # p-value
    p = (1.0 - ss.t.cdf(abs(t_stat), df)) * 2
    return p, np.mean(model1), np.mean(model2), np.std(model1), np.std(model2)

for binary in [True, False]:
    restuls_ttest = {}
    for model in ["DirectRanker",  "ListNet*", "ListNet", "RankNet", "DirectRankerV1", "LambdaMart", "AdaRank"]:
        cur_dict = {}
        runNum = "_runNum0"
        if model == "LambdaMart" or model == "AdaRank":
            runNum = "_runNum1"
        for data in ["MSLR10K", "MQ2008", "MQ2007"]:
            model_file = model
            if model == "RankNet" and data == "MSLR10K":
                runNum = "_runNum99"
            elif model == "RankNet":
                runNum = "_runNum99"
            if model == "ListNet":
                runNum = "_runNum0"
            elif model == "ListNet*":
                model_file = "ListNet"
                runNum = "_runNum99"
            if os.path.isfile(f"gridsearch/{data}_{model_file}_list_of_ndcg_ttestTrue_binary{binary}{runNum}.npy"):
                cur_dict[data] = {
                    "NDCG": np.load(f"gridsearch/{data}_{model_file}_list_of_ndcg_ttestTrue_binary{binary}{runNum}.npy", allow_pickle=True),
                    "MAP": np.load(f"gridsearch/{data}_{model_file}_list_of_map_ttestTrue_binary{binary}{runNum}.npy", allow_pickle=True),
                }
            else:
                cur_dict[data] = {"NDCG":[0 for i in range(15)],"MAP":[0 for i in range(15)]}
        restuls_ttest[model] = cur_dict

    ttest_table_dict = {"Algorithm / Data": ["MSLR10K-NDCG", "MSLR10K-MAP", "MQ2008-NDCG", "MQ2008-MAP", "MQ2007-NDCG", "MQ2007-MAP"]}
    ttest_value_dict = {}
    for model1 in ["DirectRanker", "ListNet*", "ListNet", "RankNet", "LambdaMart", "AdaRank"]:
        cur_list = []
        for model2 in ["DirectRanker", "ListNet*", "ListNet", "RankNet", "LambdaMart", "AdaRank"]:
            for data in ["MSLR10K", "MQ2008", "MQ2007"]:
                for metric in ["NDCG", "MAP"]:
                    if model1 == model2:
                        mean = np.mean(restuls_ttest[model1][data][metric])
                        std = np.std(restuls_ttest[model1][data][metric])
                        cur_list.append(f"{round(mean,3)}({int(round(std,3)*1000)})")
                        ttest = [1]
                    else:
                        ttest = bengio_nadeau_test(restuls_ttest[model1][data][metric], restuls_ttest[model2][data][metric])
                    ttest_value_dict[f"{data}-{metric}-{model1}-{model2}"] = ttest[0]
        ttest_table_dict[model1] = cur_list
    print(f"Binary {binary}")
    print(pd.DataFrame(ttest_table_dict))

    # plot ttest results
    model_names = ["DirectRanker", "ListNet*", "ListNet", "RankNet", "LambdaMart", "AdaRank"]
    model_names2 = [r"RankNet$^*$", "ListNet$^*$", "ListNet", "RankNet", "LambdaMart", "AdaRank"]
    fig, ax = plt.subplots(2, 3, constrained_layout=True)
    # define colormap
    white = np.array([255/255, 255/255, 255/255, 1])
    purple = np.array([107/255, 76/255, 154/255, 1])
    blue = np.array([57/255, 106/255, 177/255, 1])
    orange = np.array([218/255, 124/255, 48/255, 1])
    red = np.array([204/255, 37/255, 41/255, 1])
    black = np.array([83 / 255, 81 / 255, 84 / 255, 1])
    viridis = mpl.colormaps['viridis'].resampled(1000)
    newcolors = viridis(np.linspace(0, 1, 10000))
    newcolors[9500:, :] = white # self correlation
    newcolors[500:9500, :] = black # NS
    newcolors[:500, :] = red # 0.05
    # newcolors[10:100, :] = orange # 0.01
    # newcolors[:10, :] = red # 0.001
    newcmp = ListedColormap(newcolors)
    for col, data in enumerate(["MSLR10K", "MQ2008", "MQ2007"]):
        for row, metric in enumerate(["NDCG", "MAP"]):
            df_list = []
            df_list_bigger = []
            for model1 in model_names:
                cur_list = []
                cur_list_bigger = []
                for model2 in model_names:
                    if np.mean(restuls_ttest[model1][data][metric]) > np.mean(restuls_ttest[model2][data][metric]):
                        cur_list_bigger.append(1)
                    else:
                        cur_list_bigger.append(0)
                    cur_list.append(ttest_value_dict[f"{data}-{metric}-{model1}-{model2}"])
                df_list.append(cur_list)
                df_list_bigger.append(cur_list_bigger)
            pc = pd.DataFrame(df_list)
            for coli, _ in enumerate(model_names):
                for rowi, _ in enumerate(model_names):
                    if rowi > coli:
                        pc[coli].loc[rowi] = 1
                    else:
                        value = round(pc[coli].loc[rowi], 3)
                        if round(pc[coli].loc[rowi], 3) == 0:
                            value = "< 0.001"

                        if df_list_bigger[coli][rowi] == 1 and round(pc[coli].loc[rowi], 3) <= 0.05:
                            ax[row][col].annotate("", xy=(coli+0.3, rowi), xytext=(coli+0.3, rowi+0.5), arrowprops=dict(arrowstyle="->", color=white))
                            ax[row][col].text(coli, rowi, value, va='center', ha='center', size=4, color="white", fontstyle='italic')
                        elif df_list_bigger[coli][rowi] == 0 and round(pc[coli].loc[rowi], 3) <= 0.05:
                            ax[row][col].annotate("", xy=(coli, rowi+0.3), xytext=(coli+0.5, rowi+0.3), arrowprops=dict(arrowstyle="->", color=white))
                            ax[row][col].text(coli, rowi, value, va='center', ha='center', size=4, color="white", fontstyle='italic')
                        else:
                            ax[row][col].text(coli, rowi, value, va='center', ha='center', size=4, color="white")
            ax[row][col].matshow(pc, cmap=newcmp, vmin=0, vmax=1)
            data_name = data
            if data == "MSLR10K":
                data_name = "MSLR-WEB10K"
            ax[row][col].set_title(f"{data_name} {metric}")
            ax[row][col].set_xticks(np.arange(len(model_names)))
            ax[row][col].set_xticklabels(model_names2, fontsize=6, rotation=45)
            ax[row][col].set_yticks(np.arange(len(model_names)))
            ax[row][col].set_yticklabels(model_names2, fontsize=6, rotation=45)

    plt.savefig(f"15cv_ttest_binary{binary}.pdf")

# get tensorflow V1 results for DirectRanker
if results["DirectRanker"]["MSLR10K"]["NDCG-mean"] < np.mean(np.load("gridsearch/MSLR10K_DirectRankerV1_list_of_ndcg.npy")):
    results["DirectRanker"]["MSLR10K"]["NDCG-mean"] = np.mean(np.load("gridsearch/MSLR10K_DirectRankerV1_list_of_ndcg.npy"))
    results["DirectRanker"]["MSLR10K"]["MAP-mean"] = np.mean(np.load("gridsearch/MSLR10K_DirectRankerV1_list_of_map.npy"))
    results["DirectRanker"]["MSLR10K"]["NDCG-std"] = np.std(np.load("gridsearch/MSLR10K_DirectRankerV1_list_of_ndcg.npy"))
    results["DirectRanker"]["MSLR10K"]["MAP-std"] = np.std(np.load("gridsearch/MSLR10K_DirectRankerV1_list_of_map.npy"))
if results["DirectRanker"]["MQ2008"]["NDCG-mean"] < np.mean(np.load("gridsearch/MQ2008_DirectRankerV1_list_of_ndcg.npy")):
    results["DirectRanker"]["MQ2008"]["NDCG-mean"] = np.mean(np.load("gridsearch/MQ2008_DirectRankerV1_list_of_ndcg.npy"))
    results["DirectRanker"]["MQ2008"]["MAP-mean"] = np.mean(np.load("gridsearch/MQ2008_DirectRankerV1_list_of_map.npy"))
    results["DirectRanker"]["MQ2008"]["NDCG-std"] = np.std(np.load("gridsearch/MQ2008_DirectRankerV1_list_of_ndcg.npy"))
    results["DirectRanker"]["MQ2008"]["MAP-std"] = np.std(np.load("gridsearch/MQ2008_DirectRankerV1_list_of_map.npy"))
if results["DirectRanker"]["MQ2007"]["NDCG-mean"] < np.mean(np.load("gridsearch/MQ2007_DirectRankerV1_list_of_ndcg.npy")):
    results["DirectRanker"]["MQ2007"]["NDCG-mean"] = np.mean(np.load("gridsearch/MQ2007_DirectRankerV1_list_of_ndcg.npy"))
    results["DirectRanker"]["MQ2007"]["MAP-mean"] = np.mean(np.load("gridsearch/MQ2007_DirectRankerV1_list_of_map.npy"))
    results["DirectRanker"]["MQ2007"]["NDCG-std"] = np.std(np.load("gridsearch/MQ2007_DirectRankerV1_list_of_ndcg.npy"))
    results["DirectRanker"]["MQ2007"]["MAP-std"] = np.std(np.load("gridsearch/MQ2007_DirectRankerV1_list_of_map.npy"))

# set ranklib results from paper
if results["RankNet"]["MSLR10K"]["NDCG-mean"] < 0.157:
    results["RankNet"]["MSLR10K"]["NDCG-mean"] = 0.157
    results["RankNet"]["MSLR10K"]["NDCG-std"] = 0.003
    results["RankNet"]["MSLR10K"]["MAP-mean"] = 0.195
    results["RankNet"]["MSLR10K"]["MAP-std"] = 0.002
if results["RankNet"]["MQ2008"]["NDCG-mean"] < 0.716:
    results["RankNet"]["MQ2008"]["NDCG-mean"] = 0.716
    results["RankNet"]["MQ2008"]["NDCG-std"] = 0.011
    results["RankNet"]["MQ2008"]["MAP-mean"] = 0.642
    results["RankNet"]["MQ2008"]["MAP-std"] = 0.010
if results["RankNet"]["MQ2007"]["NDCG-mean"] < 0.525:
    results["RankNet"]["MQ2007"]["NDCG-mean"] = 0.525
    results["RankNet"]["MQ2007"]["NDCG-std"] = 0.011
    results["RankNet"]["MQ2007"]["MAP-mean"] = 0.525
    results["RankNet"]["MQ2007"]["MAP-std"] = 0.007

if results["ListNet"]["MSLR10K"]["NDCG-mean"] < 0.157:
    results["ListNet"]["MSLR10K"]["NDCG-mean"] = 0.157
    results["ListNet"]["MSLR10K"]["NDCG-std"] = 0.003
    results["ListNet"]["MSLR10K"]["MAP-mean"] = 0.192
    results["ListNet"]["MSLR10K"]["MAP-std"] = 0.002
if results["ListNet"]["MQ2008"]["NDCG-mean"] < 0.719:
    results["ListNet"]["MQ2008"]["NDCG-mean"] = 0.719
    results["ListNet"]["MQ2008"]["NDCG-std"] = 0.010
    results["ListNet"]["MQ2008"]["MAP-mean"] = 0.647
    results["ListNet"]["MQ2008"]["MAP-std"] = 0.006
if results["ListNet"]["MQ2007"]["NDCG-mean"] < 0.526:
    results["ListNet"]["MQ2007"]["NDCG-mean"] = 0.526
    results["ListNet"]["MQ2007"]["NDCG-std"] = 0.010
    results["ListNet"]["MQ2007"]["MAP-mean"] = 0.525
    results["ListNet"]["MQ2007"]["MAP-std"] = 0.009

for model in ["LambdaMart", "AdaRank", "ES-Rank", "IESR-Rank", "IESVM-Rank"]:
    cur_dict = {}
    for data in ["MSLR10K", "MQ2008", "MQ2007"]:
        cur_dict[data] = {
            "NDCG-mean": 0,
            "MAP-mean": 0,
            "NDCG-std": 0,
            "MAP-std": 0
        }
    results[model] = cur_dict

results["LambdaMart"]["MSLR10K"]["NDCG-mean"] = 0.476
results["LambdaMart"]["MSLR10K"]["NDCG-std"] = 0.003
results["LambdaMart"]["MSLR10K"]["MAP-mean"] = 0.366
results["LambdaMart"]["MSLR10K"]["MAP-std"] = 0.003

results["LambdaMart"]["MQ2008"]["NDCG-mean"] = 0.723
results["LambdaMart"]["MQ2008"]["NDCG-std"] = 0.007
results["LambdaMart"]["MQ2008"]["MAP-mean"] = 0.624
results["LambdaMart"]["MQ2008"]["MAP-std"] = 0.006

results["LambdaMart"]["MQ2007"]["NDCG-mean"] = 0.531
results["LambdaMart"]["MQ2007"]["NDCG-std"] = 0.012
results["LambdaMart"]["MQ2007"]["MAP-mean"] = 0.510
results["LambdaMart"]["MQ2007"]["MAP-std"] = 0.011

results["AdaRank"]["MSLR10K"]["NDCG-mean"] = 0.400
results["AdaRank"]["MSLR10K"]["NDCG-std"] = 0.016
results["AdaRank"]["MSLR10K"]["MAP-mean"] = 0.322
results["AdaRank"]["MSLR10K"]["MAP-std"] = 0.010

results["AdaRank"]["MQ2008"]["NDCG-mean"] = 0.722
results["AdaRank"]["MQ2008"]["NDCG-std"] = 0.010
results["AdaRank"]["MQ2008"]["MAP-mean"] = 0.653
results["AdaRank"]["MQ2008"]["MAP-std"] = 0.009

results["AdaRank"]["MQ2007"]["NDCG-mean"] = 0.526
results["AdaRank"]["MQ2007"]["NDCG-std"] = 0.010
results["AdaRank"]["MQ2007"]["MAP-mean"] = 0.527
results["AdaRank"]["MQ2007"]["MAP-std"] = 0.010

results["ES-Rank"]["MSLR10K"]["NDCG-mean"] = 0.382
results["ES-Rank"]["MSLR10K"]["MAP-mean"] = 0.570
results["ES-Rank"]["MQ2008"]["NDCG-mean"] = 0.507
results["ES-Rank"]["MQ2008"]["MAP-mean"] = 0.483
results["ES-Rank"]["MQ2007"]["NDCG-mean"] = 0.451
results["ES-Rank"]["MQ2007"]["MAP-mean"] = 0.470

results["IESR-Rank"]["MSLR10K"]["NDCG-mean"] = 0.415
results["IESR-Rank"]["MSLR10K"]["MAP-mean"] = 0.603
results["IESR-Rank"]["MQ2008"]["NDCG-mean"] = 0.517
results["IESR-Rank"]["MQ2008"]["MAP-mean"] = 0.494
results["IESR-Rank"]["MQ2007"]["NDCG-mean"] = 0.455
results["IESR-Rank"]["MQ2007"]["MAP-mean"] = 0.473

results["IESVM-Rank"]["MSLR10K"]["NDCG-mean"] = 0.224
results["IESVM-Rank"]["MSLR10K"]["MAP-mean"] = 0.457
results["IESVM-Rank"]["MQ2008"]["NDCG-mean"] = 0.498
results["IESVM-Rank"]["MQ2008"]["MAP-mean"] = 0.473
results["IESVM-Rank"]["MQ2007"]["NDCG-mean"] = 0.436
results["IESVM-Rank"]["MQ2007"]["MAP-mean"] = 0.456

print(results)

model_names = ["DirectRanker", "RankNet", "ListNet", "LambdaMart", "AdaRank", "ES-Rank", "IESR-Rank", "IESVM-Rank"]

# performe Friedman Nemenyi Test
data_list = []
std_list = []
for data in ["MSLR10K", "MQ2008", "MQ2007"]:
    for metric in ["NDCG", "MAP"]:
        cur_row = []
        cur_row_std = []
        for model in model_names:
            cur_row.append(results[model][data][f"{metric}-mean"])
            cur_row_std.append(results[model][data][f"{metric}-std"])
        data_list.append(cur_row)
        std_list.append(cur_row_std)
data_list = np.array(data_list)
std_list = np.array(std_list)

print("Table-Data: ", data_list.T, std_list.T)

friedman_result = ss.friedmanchisquare(*data_list.T)

print(friedman_result[1])

model_names = [r"RankNet^$*$", "RankNet", "ListNet", "LambdaMart", "AdaRank", "ES-Rank", "IESR-Rank", "IESVM-Rank"]
# if pvalue < 0.05 we can reject the null hypothesis and do a post hoc order test
if friedman_result[1] < 0.05:
    pc = sp.posthoc_nemenyi_friedman(data_list)
    print(pc)
    fig, ax_kwargs = plt.subplots(figsize=(3.3, 2.5), constrained_layout=True)
    heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
    ax, cbar = sp.sign_plot(pc, **heatmap_args)
    ax.set_xticks(np.arange(len(model_names))+0.5)
    ax.set_xticklabels(model_names, fontsize=5, rotation=45)
    ax.set_yticks(np.arange(len(model_names))+0.5)
    ax.set_yticklabels(model_names, fontsize=5, rotation=45)
    cbar.ax.tick_params(size=0, labelsize=5)
    cbar.orientation = "horizontal"
    plt.savefig('friedman_test.pdf')
