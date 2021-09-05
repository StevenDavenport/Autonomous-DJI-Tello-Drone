import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import cv2, os
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt


'''
Function = load_raw_data:
    Parameters:
        path = the path to the main data folder, containing the
               sub folders of classes.
    Returns:
        Two numpy arrays of the raw data, and corresponding labels.
        The arrays have been split for training and validation with
        a 0.8, 0.2 split.
    Algorithm:
        -Loop through the main folder.
        -Loop through the sub folders.
        -Collect the variables.
'''
def load_raw_data(path):
    X = []
    y = []
    for folder_name in os.listdir(path):
        for picture in os.listdir(path + '/' + folder_name):
            X.append(process(cv2.imread(path + '/' + folder_name + '/' + picture)))
            y.append(folder_name)
    return train_test_split(np.array(X), np.array(y), test_size=0.2)


'''
Function = load_data:
    Parameters:
        path = the path to the main data folder, containing the
               sub folders of classes.
        num_classes = the number of classes within the dataset.
    Returns:
        x_train = collection of training images
        y_train = corresponding labels to x_train
        x_test = collection of validation images
        y_test = corresponding labels to x_test
    Algorithm:
        -Load split data and labels.
        -Convert to type float32.
        -reshape dataset ready for the neural network.
        -onehot encode the labels. (y)
        -return data and labels.
'''
def load_data(path, num_classes):
    x_train, x_test, y_train, y_test = load_raw_data(path)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train.reshape(len(x_train), 50, 50, 1)
    x_test = x_test.reshape(len(x_test), 50, 50, 1)
    y_test = quantify_labels(y_test)
    y_train = quantify_labels(y_train)
    return x_train, y_train, x_test, y_test


'''
Function = process:
    Parameters:
        frame = the array of pixel values to be processed.
    Returns: 
        preprocessed frame.
    Algorithm:
        -convert to gray scale.
        -resize the image
        -calculte the mean
        -use the mean to standardize the data
'''
def process(frame):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayFrame = cv2.resize(grayFrame, (50, 50))
    mean, std = grayFrame.mean(), grayFrame.std()
    grayFrame = (grayFrame - mean) / std
    return grayFrame


'''
Function = quantify_labels:
    Parameters:
        labels = the labels to be converted.
    Returns:
        encoded_labels = the one hot encoded labels.
    Algorithm:
        -create label encoder and fit with each unique label.
        -flatten it and onehot encode it.
        -convert it to array and return.
'''
def quantify_labels(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(np.unique(labels))
    encoded_labels = label_encoder.transform(labels)
    encoded_labels = np.reshape(encoded_labels, (-1, 1))
    encoder = OneHotEncoder()
    encoder.fit(encoded_labels)
    encoded_labels = encoder.transform(encoded_labels)
    return encoded_labels.toarray()


'''
Function build_model:
    Returns:
        model = a neural network.
    Algorithm:
        -add layers sequentially
        -return the model
'''
def build_model():
    model = models.Sequential()

    # first block
    model.add(layers.Conv2D(32, (3,3), padding='same', input_shape=(50,50,1), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))

    # second block
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))

    # third block
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.4))

    # flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation='softmax'))
    return model


'''
Function build_model:
    Returns:
        model = a neural network.
    Algorithm:
        -add layers sequentially
        -return the model
'''
def build_baby_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), padding='same', input_shape=(100,100,1), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation='softmax'))
    return model


'''
Function build_model:
    Returns:
        model = a neural network.
    Algorithm:
        -add layers sequentially
        -return the model
'''
def build_baby_model2():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), padding='same', input_shape=(100, 100, 1), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='softmax'))
    return model