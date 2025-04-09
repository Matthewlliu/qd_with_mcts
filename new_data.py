import pandas as pd

df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")

print(df["Prompt"].head())
print(df["Answer"].head())
print(df.columns)