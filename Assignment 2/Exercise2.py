import pandas as pd
from collections import Counter


df = pd.read_csv("data_final.csv", sep="\t")
df["count"] = df["sample"].apply(lambda x: len(x.split(" ")))
df["count_water"] = df["sample"].apply(lambda x: Counter(x.split(" "))["water"])

print("mean number of words in sample", str(df["count"].mean()))
print("standard deviation of words in sample", str(df["count"].std()))
print("mean number of times word water appeared in each sample", str(df["count_water"].mean()))
print("mean number of times word water appeared in each sample", str(df["count_water"].std()))
df["Nora"] = df["Nora"].apply(lambda x: x.title())
df_agree = df[df["Nora"] == df["Eva"]]

print("percentage of agreement:", str(len(df_agree)/len(df)))
df_disagree = df[df["Nora"] != df["Eva"]]
df_disagree.to_csv("data_disagree.csv", sep = "\t")
