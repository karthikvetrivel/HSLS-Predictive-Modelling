import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

X = pd.read_csv("data/processed/baseline_features_final.csv", index_col=0)
y = pd.read_csv("data/processed/output_columns_final.csv", index_col=0) 

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
from sklearn.model_selection import KFold

kfold = KFold(10, True, 1)
cross_acc = []

for train_index, test_index in kfold.split(X.head(8000)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = Sequential()
    model.add(Dense(20, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        
    # model.fit(X_train, y_train, epochs=500, validation_split=0.2, batch_size=1024)
    model.fit(X_train, y_train, validation_split=0.2, epochs=100)
    val_loss, val_acc = model.evaluate(X_test, y_test)
    
    cross_acc.append(val_acc)

cross_accuracies = pd.DataFrame(cross_acc)
cross_accuracies.to_csv("figures/kFold_crossvalidation_accuracies.csv")

# X_train, X_test, y_train, y_test = train_test_split(baseline_features, output_columns, test_size = 0.2)
# X_train = np.asarray(X_train.head(8000))
# X_test = np.asarray(X_test.head(8000))
# y_train = np.asarray(y_train.astype(int)[0:8000])
# y_test = np.asarray(y_test.astype(int)[0:8000])

# model = Sequential()
# model.add(Dense(20, input_dim=baseline_features.shape[1], activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
# model.fit(X_train, y_train, epochs=500, validation_split=0.2, batch_size=1024)
# val_loss, val_acc = model.evaluate(X_test, y_test)

