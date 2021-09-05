'''
This file was used to collect data, using the tello drone's onbaord
camera, to create training data for the gesture recognition cnn.
The collected files can be found in './data' where there are
4 folders of pictures which are 4 classes for the network to 
classify.
''' 


from djitellopy import Tello 
import cv2, os, time


'''
Function = get_file_id:
    Parameters:
        folder = the folder to store the collected data.
    Returns:
        full_directory = the full directory of the frame to be saved, 
                         including it's name (file_id).
    Algorithm:
        -search through the folder and create a unique file name.
'''
def get_file_id(folder, directory='C:/Users/Steven/Desktop/AutoTelloTest/gesture/data/',file_id=0, file_type='.png'):
    full_directory = ''
    while True:
        if os.path.isfile(directory + folder + '/' + folder +str(file_id) + file_type):
            file_id +=1
        else:
            full_directory = directory + folder + '/' + folder + str(file_id) + file_type
            break
    return full_directory


'''
Function = take_picture:
    Parameters:
        folder = the folder to store the collected data.
    Algorithm:
        -connect to tello drone.
        -display video feed.
        -take 5 pictures over a small peroid of time.
        -save the 5 picture to the correct folders.
'''
def take_picture(folder, directory='C:/Users/Steven/Desktop/AutoTelloTest/gesture/data/',file_id=0, file_type='.png'):
    tello = Tello()
    tello.connect()
    tello.streamoff()
    tello.streamon()
    time.sleep(2)
    for x in range(25):
        full_directory = get_file_id(folder)
        frame = tello.get_frame_read().frame
        cv2.imshow('tello feed', frame)
        cv2.imwrite(full_directory, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        time.sleep(1)
    cv2.destroyAllWindows


if __name__ == '__main__':
     take_picture('takeoff')


    