import pandas as pd

file_name = "grid_lhs_constrained_final_choice.csv"

df = pd.read_csv(file_name)

basic_stats = df.describe().T
basic_stats["skew"] = df.skew()

full_table = basic_stats[["mean", "min", "max", "std"]].to_string()
print(full_table)

basic_stats.to_csv(file_name.split(".")[0]+"_basic_stats.csv", sep=",", index=True)