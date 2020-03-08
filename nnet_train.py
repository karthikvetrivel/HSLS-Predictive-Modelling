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
cross_results = []

nodes_val = [10, 17, 20, 25]

for val in nodes_val:
    for train_index, test_index in kfold.split(X.head(5000)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = Sequential()
        model.add(Dense(val, input_dim=X.shape[1], activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(5, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',
                      optimizer='adam', metrics=['accuracy'])

        # model.fit(X_train, y_train, epochs=500, validation_split=0.2, batch_size=1024)
        model.fit(X_train, y_train, validation_split=0.2, epochs=50)
        val_loss, val_acc = model.evaluate(X_test, y_test)

    cross_results.append(val_acc)


joint_data = list(zip(nodes_val, cross_results))  
cross_results_final = pd.DataFrame(joint_data, columns=['NumNodes', 'Acc'])
cross_results_final.head()
cross_results_final.to_csv("figures/kFold_crossvalidation_nodes.csv", index=False)

# X_train, X_test, y_train, y_test = train_test_split(baseline_features, output_columns, test_size = 0.2)
# X_train = np.asarray(X_train.head(8000))
# X_test = np.asarray(X_test.head(8000))
# y_train = np.asarray(y_train.astype(int)[0:8000])
# y_test = np.asarray(y_test.astype(int)[0:8000])
w
# model = Sequential()
# model.add(Dense(20, input_dim=baseline_features.shape[1], activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=500, validation_split=0.2, batch_size=1024)
# val_loss, val_acc = model.evaluate(X_test, y_test)
