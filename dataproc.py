import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mpl

student_data = pd.read_csv("data/HSLS_2016_v1_0_CSV_Datasets/hsls_16_student_v1_0.csv");
student_data.head()


output_col = student_data[['X3CLGANDWORK', 'S3CLASSES', 'S3WORK'
, 'S3APPRENTICE', 'X4EVRAPPCLG', 'S3CLGAPPNUM', 'S4EVERAPPLY', 'S4CHOICEAPP']]
output_col.head()

corr = output_col.corr()
mpl.figure(figsize=(12,10), dpi=50)
sns.heatmap(corr, 
    xticklabels=corr.columns, 
    yticklabels=corr.columns)
mpl.savefig(fname="figures/dataproc_correlation_outputs.png", dpi=300)

# Strong correlation between X3CLGANDWORK & S3Classes and S3Work and S3Apprentice
# Nearly 1

output_col.describe()
