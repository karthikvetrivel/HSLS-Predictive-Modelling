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
output_col.head()
df = output_col.isnull().sum() 
df.to_csv("figures/meanimputation_output_null.csv")

# TO-DO: Find pattern between output and input variable. 
output_cols_removed = student_data.loc[:,~student_data.columns.isin(output_cols)]
output_cols_removed_count = output_cols_removed.isna().sum()
df = output_cols_removed_count / len(output_cols_removed)
predictor_col = output_cols_removed.fillna(output_cols_removed.median()) 

# TO-DO: Find variable w/ 'missing-ness' > 10%
# TO-DO: Find potential correlation.


# Proportion of Data missing per column
df.to_csv("figures/meanimputation_predictor_nullproportion.csv")

