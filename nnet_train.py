import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import argparse


baseline_features = pd.read_csv("data/processed/baseline_features.csv", index_col=0)
output_columns = pd.read_csv("data/processed/output_columns.csv", index_col=0) 

# 23503 x 1470 columns
baseline_features.head()
# 23503 x 6 columns
output_columns.head()

# TO-DO, build a model that trains on each individual output column




# parser = argparse.ArgumentParser()
# parser.add_argument('--output_col', type=str, help='name of output', default=None)
# args = parser.parse_args()

# Merge into a baseline and output into a single column
df = pd.merge(baseline_features, output_columns, on='STU_ID')

# Remove rows w/ NaN values in the output column
df = df.dropna(axis=0, subset=output_columns.columns)


baseline_features_final = df[output_columns.columns]
baseline_features_final.drop(['STU_ID'], axis=1)
output_columns_final = df[baseline_features.columns] 
output_columns_final.drop(['STU_ID'],axis=1)

output_columns_final.head()

