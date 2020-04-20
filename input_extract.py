import pandas as pd
import numpy as np

student_data = pd.read_csv(
    "data/HSLS_2016_v1_0_CSV_Datasets/hsls_16_student_v1_0.csv", na_values=[-9, -8, -5, -7, -4, -3])
student_data.head()

# -9 = No Unit Response
# -8 = Missing
# -5 = Supressed
# -7 = Skipped
# -4 = Question not adminstered
# -3 = Carry through missing
# Sort by the first gathered predictor variables

# Getting columns that start with certain keywords
filter_col = [col for col in student_data if col.startswith('S1') or col.startswith('X1') or col.startswith('A1') or col.startswith(
    'C1')or col.startswith('STU_ID')]
X = student_data[filter_col]

updated_predictor_col_count = X.isna().sum()
df = updated_predictor_col_count / len(X)
df.to_csv("figures/missingness_imputation_missingness_per_predictor.csv")

# Only take columns with less than 15% of the data missing
X = X[X.columns[X.isnull().mean() < 0.15]]

# Median Imputation
updated_predictor_col = X.fillna(
    X.median())
updated_predictor_col.head()


# Export to CSV file
updated_predictor_col.to_csv("data/processed/baseline_features.csv", index=False)

