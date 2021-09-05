import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import cv2, os
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

from utils import *

'''
Function = build_and_train_model:
    This function uses the utils file to load, 
    transform dataset, then build and train the model.
    That model is then saved.
'''
def build_and_train_model():
    epochs = 30
    num_classes = 5

    x_train, y_train, x_test, y_test = load_data('gdrive/My Drive/Year3/Project/gesture/data', num_classes)
    model = build_baby_model2(x_train, num_classes)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    batch_size = 16
    model = build_baby_model(x_train, num_classes)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    model.save_weights('gdrive/My Drive/Year3/Project/model/gesture_model3.h5')
    print('\nTest score: ', score[0])
    print('\nTest accuracy: ', score[1])

if __name__ == '__main__':
    build_and_train_model()
