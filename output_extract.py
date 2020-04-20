import pandas as pd 

student_data = pd.read_csv("data/HSLS_2016_v1_0_CSV_Datasets/hsls_16_student_v1_0.csv", na_values=[-9, -8, -5, -7, -4, -3]);
student_data.head()

# -9 = No Unit Response
# -8 = Missing
# -5 = Supressed
# -7 = Skipped
# -4 = Question not adminstered
# -3 = Carry through missing

output_cols = ['STU_ID', 'X3CLGANDWORK', 'S3CLASSES', 'S3WORK'
, 'S3APPRENTICE', 'X4EVRAPPCLG', 'S3CLGAPPNUM']
output_col = student_data[output_cols]
output_col

df = output_col.isnull().sum() 
df.to_csv("figures/output_extract_missingness_per_output.csv")

output_col.to_csv("data/processed/output_columns.csv", index=False)