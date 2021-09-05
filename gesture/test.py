from control import GestureControl
import cv2, time
import matplotlib.pyplot as plt

gc = GestureControl()
frame = cv2.imread('C:/Users/Steven/Desktop/AutoTello/gesture/data/takeoff/takeoff21.png')
#plt.imshow(frame)
#plt.show()
print(gc.predict(frame))








'''
THIS FILE IS JUST FOR TESTING THE GESTURE CONTROL CLASS, AND THE ACCURACY OF THE MODEL.
IT IS NOT SIGNIFICANT AND REALISTICALLY IF THE PROJECT WAS BEING RELEASED, IT COULD BE DELETED.
'''