import pandas as pd
import numpy as np

student_data = pd.read_csv("data/HSLS_2016_v1_0_CSV_Datasets/hsls_16_student_v1_0.csv", na_values=[-9, -8, -5, -7, -4, -3]);
student_data.head()

# -9 = No Unit Response
# -8 = Missing
# -5 = Supressed
# -7 = Skipped
# -4 = Question not adminstered
# -3 = Carry through missing
# Sort by the first gathered predictor variables

# Getting columns that start with certain keywords
filter_col = [col for col in student_data if col.startswith('S1') or col.startswith('X1') or col.startswith('A1') or col.startswith('C1') or col.startswith('S2') or col.startswith('X2') or col.startswith('A2') or col.startswith('C2') or col.startswith('STU_ID')]
updated_predictor_col = student_data[filter_col] 

updated_predictor_col_count = updated_predictor_col.isna().sum()
df = updated_predictor_col_count / len(updated_predictor_col)
df.to_csv("figures/missingness_imputation_missingness_per_predictor.csv")

# Median Imputation
updated_predictor_col = updated_predictor_col.fillna(updated_predictor_col.median()) 
updated_predictor_col.head()

# Removing columsn with ALL missing values
final_predictor_col = updated_predictor_col.dropna(axis=1)

# Export to CSV file
final_predictor_col.to_csv("data/processed/baseline_features.csv")








