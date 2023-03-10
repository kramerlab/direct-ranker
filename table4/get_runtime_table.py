import pandas as pd
import numpy as np


result_table = {
    "Time / Model": ["LambdaMart", "AdaRank", "ListNet", "RankNet", "DirectRanker"],
    "read data [s]": [],
    "train model [s]": []
    }

for m in ["LambdaMart", "AdaRank", "ListNet", "RankNet", "DirectRanker"]:
    df = pd.read_csv(f"output/{m}_time.txt", header=None)
    result_table["read data [s]"].append(f"{round(np.mean(df[0]),3)}({int(round(np.std(df[0]),3)*1000)})")
    result_table["train model [s]"].append(f"{round(np.mean(df[1]),3)}({int(round(np.std(df[1]),3)*1000)})")
print(result_table)
print(pd.DataFrame(result_table).to_latex())
