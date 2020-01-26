import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np


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

# Splitting the data set into the Training and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(baseline_features_final, output_columns_final, test_size = 0.2)
X_train = X_train.head(2000)
X_test = X_test.head(2000)
y_train = y_train.astype(int)[0:2000]
y_test = y_test.astype(int)[0:2000]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))

feature_columns = [tf.feature_column.numeric_column('x', shape=X_train_scaled.shape[1:])]		

estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[300, 100],
    dropout=0.3, 
    n_classes = 10,
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
        name='Adam'
    ),
    model_dir = None)

train_input = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": X_train_scaled},
    y=y_train.values,
    batch_size=50,
    shuffle=False,
    num_epochs=None)

estimator.train(input_fn = train_input, steps=5000)


eval_input = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": X_test_scaled},
    y=y_test.values, 
    shuffle=False,
    batch_size=X_test_scaled.shape[0],
    num_epochs=1)
estimator.evaluate(eval_input,steps=None) 