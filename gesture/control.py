from tensorflow.keras import datasets, layers, models, optimizers, regularizers
import numpy as np
import cv2, time, threading


'''
Class = GestureControl()
    Loads in a CNN that will classify gestures.
    All methods are used to preprocess frames, to be ran
    through the network. 
'''
class GestureControl():
    def __init__(self):
        self.model = self.build_model()
        self.model.load_weights('C:/Users/Steven/Desktop/AutoTello/gesture/model/gesture_model3.h5')
        self.prediction_thread = threading.Thread(target=self.run)
        self.thread_running = False
        self.frame = None
        self.gesture_prediction = 'none'


    '''
    Function = predict:
        Returns:
            result = string containing information on the movement prediction
                     based on the input frame.
        Algorithm:
            -preprocess the frame.
            -predict the movement.
            -decode the prediction.
            -return the prediction as a string.
    '''
    def predict(self):
        preprocessed_frame = self.preprocessed(self.frame)
        preprocessed_frame = preprocessed_frame.astype('float32')
        prediction = self.model.predict(preprocessed_frame)
        prediction = prediction.astype('int')
        prediction = np.array_str(prediction)
        result = self.decode(prediction)
        return result
    

    '''
    Function = decode:
        Parameters:
            prediction = the raw prediction outputted from the model.
        Returns:
            result = string containing information on the movement prediction
                     based on the input frame.
        Algorithm:
            -for each possible output from the cnn, 
                return a different string.
    '''
    def decode(self, prediction):
        result = ''
        if '1 0 0 0' in prediction:
            result = 'down'
        elif '0 1 0 0' in prediction:
            result = 'land'
        elif '0 0 1 0' in prediction:
            result = 'takeoff'
        elif '0 0 0 1' in prediction:
            result = 'up'
        else:
            result = 'none'
        return result


    '''
    Function = preprocessed:
        Parameters:
            frame = the raw pixel values of the frame to be processed.
        Returns:
            grayFrame = the fully processed frame, ready for CNN.
        Algorithm:
            -convert to grayscale.
            -convert to array.
            -convert to type float32.
            -find the mean and standardize .
            -resize.
            -reshape to input size.
            -return the result.
    '''
    def preprocessed(self, frame):
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayFrame = np.array(grayFrame)
        grayFrame.astype('float32')
        mean, std = grayFrame.mean(), grayFrame.std()
        grayFrame = (grayFrame - mean) / std
        grayFrame = cv2.resize(grayFrame, (50, 50))
        grayFrame = grayFrame.reshape(1, 50, 50, 1)
        return grayFrame


    '''
    Function = build_model:
        Returns:
            model = the model used for gesture classification.
        Algorithm:
            -add each layer at a time to
                a sequential model.
            -return the model.
    '''
    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3,3), padding='same', input_shape=(50, 50, 1), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(4, activation='softmax'))
        return model


    '''
    Function = set_frame:
        Parameters:
            frame = the raw frame's pixel values 
                    taken from the drone.
        Algorithm:
            -save the frame to member variable.
            -if cnn thread isn't running, start it. 
    '''
    def set_frame(self, frame):
        self.frame = frame
        if not self.thread_running:
            self.prediction_thread.start()
            self.thread_running = True
    

    '''
    Function = get_gesture_prediction:
        Returns:
            The stored gesture prediction (string)
        Algorithm:
            -if there is a gesture prediction stored,
                return it to caller.
    '''
    def get_gesture_prediction(self):
        if self.gesture_prediction != 'none':
            x = self.gesture_prediction
            self.gesture_prediction = 'none'
            return x
        else:
            return self.gesture_prediction


    '''
    Function = run:
        Algorithm:
            -the target of the cnn prediction thread.
            - predicts a frame every 1 second. 
    '''
    def run(self):
        while True:
            time.sleep(1)
            self.gesture_prediction = self.predict()


    '''
    Function = join_thread:
        Algorithm:
            -join the cnn prediction thread.
    '''
    def join_thread(self):
        self.prediction_thread.join()
