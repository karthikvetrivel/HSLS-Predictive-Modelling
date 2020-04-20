from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

X = pd.read_csv("data/processed/baseline_features_final.csv")
y = pd.read_csv("data/processed/output_columns_final.csv")
X.drop('X1PAREDEX_IM', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
opt = tf.keras.optimizers.Adam(lr=0.00001, decay=1e-6)

model = Sequential()
model.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer=opt)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

model.fit(x=X_train, y=y_train, epochs=50,
          validation_data=(X_test, y_test), verbose=1)
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

predictions = model.predict_classes(X_test)
print(classification_report(y_test, predictions))
