import pandas as pd

baseline_features = pd.read_csv("data/processed/baseline_features.csv", index_col=0)
output_columns = pd.read_csv("data/processed/output_columns.csv", index_col=0) 

# 23503 x 1470 columns
baseline_features.head()
# 23503 x 6 columns
output_columns.head()

# Specific column to be tested on.
main_output_column = output_columns[["S3CLASSES", "STU_ID"]]

# Merge into a baseline and output into a single column
df = pd.merge(baseline_features, main_output_column, on='STU_ID')

# Remove rows w/ NaN values in the output column
df = df.dropna(axis=0, subset=main_output_column.columns)

# Create the x and y columns
baseline_features_final = df[baseline_features.columns]
baseline_features_final = baseline_features_final.drop(['STU_ID'], axis=1)
output_columns_final = df[main_output_column.columns] 
output_columns_final = output_columns_final.drop(['STU_ID'],axis=1)

output_columns_final["S3CLASSES"] = (output_columns_final["S3CLASSES"].astype('int')) - 1

output_columns_final.to_csv("data/processed/output_columns_final.csv")
baseline_features_final.to_csv("data/processed/baseline_features_final.csv")