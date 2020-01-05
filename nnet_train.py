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

# Specific column to be tested on.
main_output_column = output_columns[['S3CLASSES', 'STU_ID']]

# Merge into a baseline and output into a single column
df = pd.merge(baseline_features, main_output_column, on='STU_ID')

# Remove rows w/ NaN values in the output column
df = df.dropna(axis=0, subset=main_output_column.columns)

# Create the x and y columns
baseline_features_final = df[main_output_column.columns]
baseline_features_final = baseline_features_final.drop(['STU_ID'], axis=1)
output_columns_final = df[baseline_features.columns] 
output_columns_final = output_columns_final.drop(['STU_ID'],axis=1)

# Splitting the data set into the Training and Testing set
import tensorflow as tf
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(baseline_features_final, output_columns_final, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set
classifier.fit(X_train, y_train.values.tolist(), batch_size = 10, nb_epoch = 300)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)