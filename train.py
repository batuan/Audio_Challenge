import numpy as np

from model import Resnet1D
import argparse
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from sklearn.model_selection import train_test_split


def get_model(timeseries, nfeatures, nclass):
    model = Sequential()
    model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(timeseries, nfeatures)))
    model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(units=nclass, activation='softmax'))
    return model


if __name__ == "__main__":
    data = np.load('./train_imgs.npy')
    label = np.load('./train_labels.npy')
    label = label.reshape((len(label), 1))
    model = get_model(data.shape[1], data.shape[2], 1)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=2018)
    # print(y_test)

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()

    batch_size = 1024
    nb_epochs = 10

    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_test, y_test))