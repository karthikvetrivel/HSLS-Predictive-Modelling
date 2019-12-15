import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


baseline_features = pd.read_csv("data/processed/baseline_features.csv", index_col=0)
output_columns = pd.read_csv("data/processed/output_columns.csv", index_col=0) 

baseline_features.head()
output_columns.head()

# Merge into a baseline and output into a single column
df = pd.merge(baseline_features, output_columns, on='STU_ID')


df
# Remove rows w/ NaN values in the output column
df = df.dropna(axis=0, subset=output_columns.columns)
df

baseline_features_final = df[output_columns.columns]
baseline_features_final.drop(['STU_ID'], axis=1)
output_columns.final = df[baseline_features.columns] 
output_columns.final.drop(['STU_ID'],axis=1)

output_columns.final.head()

