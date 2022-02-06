import numpy as np
import argparse
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras as keras
import os
import keras_resnet.models
import argparse

parser = argparse.ArgumentParser(description='Config Model')
parser.add_argument('--model-mode', metavar='N', type=int, default=4,
                    help='1 for LSTM, 2 for CNN, 3 for ResNet')
parser.add_argument('--batch', type=int, default=4096,
                    help='batch size)')
parser.add_argument('--nepoch', type=int, default=20,
                    help='batch size)')
parser.add_argument('--save-name', type=str, default='tuan.h5',
                    help='model save name')
parser.add_argument('--enhance', type=bool, default=False,
                    help='use external data or not)')
args = parser.parse_args()


def set_gpu(devide=1):
  # limit memory for GPU
  os.environ["CUDA_VISIBLE_DEVICES"]=str(devide)
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)

def get_model_LSTM(timeseries, nfeatures, nclass):
    model = Sequential()
    model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(timeseries, nfeatures)))
    model.add(LSTM(units=128, dropout=0.1, recurrent_dropout=0.1, return_sequences=False))
    model.add(Dense(units=nclass, activation='softmax'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nclass, activation='softmax'))
    return model


def get_fully_connected(timeseries, nfeatures, nclass):
  model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(timeseries, nfeatures)),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        
        # 3rd dense layer
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),

        # 4th dense layer
        keras.layers.Dense(64, activation='relu'),
        # output layer
        keras.layers.Dense(nclass, activation='softmax')
    ])
  return model


def get_model_CNN(timeseries, nfeatures, nclass):
    model = Sequential()
    input_shape=(timeseries, nfeatures, 1)
    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,  padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,  padding='same'))

    # 2nd conv layer
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu',  padding='same'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(128, activation='relu'))

    # output layer
    model.add(keras.layers.Dense(nclass, activation='softmax'))

    return model
  
def get_model_ResNet(data_shape, nbclass):
    shape, classes = (data_shape[1], data_shape[2], 1), nbclass
    x = keras.layers.Input(shape)
    model = keras_resnet.models.ResNet18(x, classes=classes)
    return model

def test_model(model):
  X_test, y_test = np.load('./data_train_test/test_imgs.npy'), np.load('./data_train_test/test_labels.npy')
  data_shape = X_test.shape
  
  if args.model_mode not in [0,3]:
    X_test = X_test.reshape((data_shape[0], data_shape[1], data_shape[2], 1))
  y_test = tf.keras.utils.to_categorical(y_test)

  score = model.evaluate(X_test, y_test, verbose=0)
  print(score)


def get_data_train_test(enhance=True, model_mode=1):
  data = np.load('./data_train_test/train_imgs.npy')
  label = np.load('./data_train_test/train_labels.npy')
  
  if enhance:
    new_data = np.load('./data_train_test/train2_imgs.npy')
    new_label = np.load('./data_train_test/train2_labels.npy')
    data = np.concatenate([data, new_data])
    label = np.concatenate([label, new_label])

  if model_mode not in [0,3]:
    data_shape = data.shape
    data = data.reshape((data_shape[0], data_shape[1], data_shape[2], 1))

  print("data shape is {}".format(data.shape))
  label = tf.keras.utils.to_categorical(label) #label.reshape((len(label), 1))
  X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.3, random_state=42)
  return X_train, X_val, y_train, y_val

 
if __name__ == "__main__":
    set_gpu()
    model_mode = args.model_mode
    enhance = args.enhance
    X_train, X_val, y_train, y_val = get_data_train_test(enhance, model_mode)

    if model_mode == 0:
      model = get_fully_connected(X_train.shape[1], X_train.shape[2], 2)
    elif model_mode == 1:
      model = get_model_LSTM(X_train.shape[1], X_train.shape[2], 2)
    elif model_mode == 2:
      model = get_model_CNN(X_train.shape[1], X_train.shape[2], 2)
    elif model_mode == 3:
      model = get_model_ResNet(X_train.shape, 2)

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()

    batch_size = args.batch
    nb_epochs = args.nepoch

    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_val, y_val))

    test_model(model)  
    model.save(args.save_name)