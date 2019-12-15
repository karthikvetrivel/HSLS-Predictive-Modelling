import pandas as pd

student_data = pd.read_csv("data/HSLS_2016_v1_0_CSV_Datasets/hsls_16_student_v1_0.csv", na_values=[-9, -8, -5, -7, -4, -3]);
student_data.head()

# -9 = No Unit Response
# -8 = Missing
# -5 = Supressed
# -7 = Skipped
# -4 = Question not adminstered
# -3 = Carry through missing

output_cols = ['X3CLGANDWORK', 'S3CLASSES', 'S3WORK'
, 'S3APPRENTICE', 'X4EVRAPPCLG', 'S3CLGAPPNUM']
output_col = student_data[output_cols]
# Proportion of Data missing per column
output_cols_removed = student_data.loc[:,~student_data.columns.isin(output_cols)]
output_cols_removed_count = output_cols_removed.isna().sum()
df = output_cols_removed_count / len(output_cols_removed)
df.to_csv("figures/missingness_imputation_missingness_per_predictor.csv")


# Get output_col_removed imputed with the median values
predictor_col = output_cols_removed.fillna(output_cols_removed.median()) 

# Sort by the first gathered predictor variables
filter_col = [col for col in predictor_col if col.startswith('S1') or col.startswith('X1') or col.startswith('A1') or col.startswith('C1') or col.startswith('S2') or col.startswith('X2') or col.startswith('A2') or col.startswith('C2') or col.startswith('STU_ID')]
updated_predictor_col = student_data[filter_col] 

updated_predictor_col.to_csv("data/processed/baseline_features.csv")








