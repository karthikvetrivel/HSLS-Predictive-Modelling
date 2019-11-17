import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mpl

student_data = pd.read_csv("data/HSLS_2016_v1_0_CSV_Datasets/hsls_16_student_v1_0.csv", na_values=);
student_data.head()


output_col = student_data[['X3CLGANDWORK', 'S3CLASSES', 'S3WORK'
, 'S3APPRENTICE', 'X4EVRAPPCLG', 'S3CLGAPPNUM', 'S4EVERAPPLY', 'S4CHOICEAPP']]

corr = output_col.corr()
sns.heatmap(corr, 
    xticklabels=corr.columns, 
    yticklabels=corr.columns)

# Strong correlation between X3CLGANDWORK & S3Classes and S3Work and S3Apprentice
# Nearly 1


output_col.describe()