import numpy as np
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt


def get_ranklib():
    # get ranklib results
    for num, algo in zip([1, 7, 6, 3], ["LambdaMart", "AdaRank", "ListNet", "RankNet"]):
        cur_dict = {}
        for pathname in zip(["MQ2007", "MQ2008", "MSLR-WEB10K"]):
            MAP_list = [0, 0, 0, 0, 0]
            NDCG_list = [0, 0, 0, 0, 0]
            for idx_fold, fold in enumerate(["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]):
                for filename in os.listdir(pathname + "/"):
                    if algo not in filename:
                        continue
                    if fold not in filename:
                        continue
                    with open(pathname + "/" + filename, "r") as file:
                        for line in file:
                            if "Avg." in line:
                                print((line.replace("\n","").split()[2]))
                                if "MAP" in filename:
                                    if best_map[idx_fold] < float(line.replace("\n","").split()[2]):
                                        best_map[idx_fold] = float(line.replace("\n","").split()[2])         
                                        map_para[fold] = filename
                                else:
                                    if best_ndcg[idx_fold] < float(line.replace("\n","").split()[2]):
                                        best_ndcg[idx_fold] = float(line.replace("\n","").split()[2])         
                                        ndcg_para[fold] = filename    
            print(algo)
            print("MAP " + str(best_map))
            print("NDCG@10  " + str(best_ndcg))

# get tensorflow V2 results
results = {}
for model in ["DirectRanker", "ListNet", "RankNet"]:
    cur_dict = {}
    for data in ["MSLR10K", "MQ2008", "MQ2007"]:
        cur_dict[data] = {
            "NDCG-mean": np.mean(np.load(f"gridsearch/{data}_{model}_list_of_ndcg.npy", allow_pickle=True)),
            "NDCG-std": np.std(np.load(f"gridsearch/{data}_{model}_list_of_ndcg.npy", allow_pickle=True)),
            "MAP-mean": np.mean(np.load(f"gridsearch/{data}_{model}_list_of_map.npy", allow_pickle=True)),
            "MAP-std": np.std(np.load(f"gridsearch/{data}_{model}_list_of_map.npy", allow_pickle=True))
        }
    results[model] = cur_dict

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

# if pvalue < 0.05 we can reject the null hypothesis and do a post hoc order test
if friedman_result[1] < 0.05:
    pc = sp.posthoc_nemenyi_friedman(data_list)
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
