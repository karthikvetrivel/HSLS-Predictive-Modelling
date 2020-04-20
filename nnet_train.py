import tensorflow as tf
import pandas as pd

X = pd.read_csv("data/processed/baseline_features_final.csv", index_col=0)
y = pd.read_csv("data/processed/output_columns_final.csv", index_col=0)

from keras import backend as K
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

kfold = KFold(10, True, 1)
cross_acc = []
cross_prec = []

# nodes_val = [10, 17, 20, 25]

# for val in nodes_val:
for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = Sequential()
    model.add(Dense(20, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(5, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',
                      optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])

        # model.fit(X_train, y_train, epochs=500, validation_split=0.2, batch_size=1024)
    model.fit(X_train, y_train, validation_split=0.2, epochs=10)
    loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test)

# joint_data = list(zip(nodes_val, cross_acc))  
# cross_results_final = pd.DataFrame(joint_data, columns=['NumNodes', 'Acc'])
# cross_results_final.head()
# cross_results_final.to_csv("figures/kFold_crossvalidation_nodes.csv", index=False)

