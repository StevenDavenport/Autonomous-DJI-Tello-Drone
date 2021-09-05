from djitellopy import Tello
from threading import Thread
import os.path, cv2, time

'''
Drone should be instantiated as an object that will connect to
and control the movement of the tello drone.
This is acomplished by the drone object being sent gesture and
tracking data. 
'''
class Drone:
    def __init__(self):
        self.tello = Tello()                    # create the tello object
        self.z_velocity = 0                     # z axis velocity
        self.yaw_velocity = 0                   # yaw axis velocity
        self.z_momentum = 0                     # z axis momentum 
        self.braking = False                    # flag, current braking status
        self.record = True                      # flag, current video-recording status
        self.tracked = False                    # flag, current user-tracked status
        self.video_thread = Thread(target=self.record_video) # video thread
        self.small_x_limit = 150                # x axis limit                  
        self.large_x_limit = 350                # x axis limit                               
        self.momentum_limit = 30                # z axis momentum limit                                             
        self.z_speed = 30                       # z axis speed
        self.yaw_speed = 40                     # yaw axis speed
        self.tello_takeoff = False              # flag, current in air status
        self.setup_tello_obj()                  # connects the tello obj to tello
        self.video_thread_started = False       # flag, current video-recording status
        self.start_video_thread()               # start the video-recording thread

    # connect the tello object to the tello drone
    def setup_tello_obj(self):
        self.tello.connect()
        self.tello.streamoff()
        self.tello.streamon()

    # start the video thread and change the flag
    def start_video_thread(self):
        if not self.video_thread_started:
            self.video_thread.start()
            self.video_thread_started = True

    # stop the video thread
    def join_video_thread(self):
        self.video_thread.join()
    
    # returns a captured frame from tello
    def get_frame(self):
        return self.tello.get_frame_read().frame

    # returns the current battery level of tello
    def get_battery(self):
        return self.tello.get_battery()

    # sends the tello drone the command to launch
    def takeoff(self):
        self.tello.takeoff()

    # this function is the target of the record video thead
    def record_video(self):
        file_id = 0
        directory = './videos/'
        file_type = '.avi'
        full_directory = ''
        height, width, _ = self.get_frame().shape

        # find a file name that has not been used
        while True:
            if os.path.isfile(directory + str(file_id) + file_type):
                file_id += 1
            else:
                full_directory = directory + str(file_id) + file_type
                break
        video = cv2.VideoWriter(full_directory, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

        # frame loop, get and save the frames
        while self.record == True: 
            video.write(self.get_frame())
            time.sleep(1 / 60)
        video.release()
    
    # function that handles the gesture command and moves the drone appropriatly. 
    def gesture_control(self, gesture):
        if 'up' in gesture and self.tello_takeoff:
            self.tello.move_up(20)
        elif 'down' in gesture and self.tello_takeoff:
            self.tello.move_down(20)
        elif 'takeoff' in gesture and not self.tello_takeoff:
            self.tello.takeoff()
            self.start_video_thread()
            self.tello_takeoff = True
        elif 'land' in gesture and self.tello_takeoff:
            self.tello.land()
            self.join_video_thread
            self.tello_takeoff = False

    def make_movement_decision(self, bbox, frame):
        # variables used to deduct multidirectional velocities
        x_vector = int(frame.shape[1] / 2) - (bbox[0] + ((bbox[2] - bbox[0]) / 2))      # vector from the center screen to the center of the user
        bb_area = abs(bbox[3] - bbox[1]) * abs(bbox[2] - bbox[0])                       # rea of the users bounding box
        high_speed_forward_limit = frame.shape[0] * frame.shape[1] / 9                  # area bounds
        medium_speed_forward_limit = frame.shape[0] * frame.shape[1] / 5                # area bounds
        medium_speed_backwards_limit = frame.shape[0] * frame.shape[1] / 3              # area bounds
        high_speed_backwards_limit = frame.shape[0] * frame.shape[1] / 2                # area bounds

        # z axis movement decision
        if bb_area < high_speed_forward_limit:
            self.z_velocity = int(4*self.z_speed)
            if self.z_momentum >= 0:
                self.z_momentum += 4
            else:
                self.z_momentum = 0
                
        # z axis movement decision
        elif bb_area < medium_speed_forward_limit:
            self.z_velocity = self.z_speed
            # Has there been a shift in momentum
            if self.z_momentum >= 0: 
                self.z_momentum += 1
            else: 
                self.z_momentum = 0

        # z axis movement decision
        elif bb_area > high_speed_backwards_limit:
            self.z_velocity = -int(1.5*self.z_speed)
            # Has there been a shift in momentum
            if self.z_momentum <= 0:
                self.z_momentum -= 1.5
            else:
                self.z_momentum = 0

        # z axis movement decision
        elif bb_area > medium_speed_backwards_limit:
            self.z_velocity = -self.z_speed
            # Has there been a shift in momentum
            if self.z_momentum <= 0:
                self.z_momentum -= 1
            else:
                self.z_momentum = 0

        else:
            # is breaking required
            if self.z_momentum > self.momentum_limit:
                self.z_velocity = -self.z_speed 
            elif self.z_velocity < -self.momentum_limit:
                self.z_velocity = self.z_speed 
            else:
                self.z_velocity = 0
                self.z_momentum = 0
                
        # yaw axis movement decision
        if x_vector > self.large_x_limit:
            self.yaw_velocity = int(-self.yaw_speed*2)

        # yaw axis movement decision
        elif x_vector > self.small_x_limit:
            self.yaw_velocity = -self.yaw_speed

        # yaw axis movement decision
        elif x_vector < -self.large_x_limit:
            self.yaw_velocity = int(self.yaw_speed*2)

        # yaw axis movement decision
        elif x_vector < -self.small_x_limit:
            self.yaw_velocity = self.yaw_speed

        # yaw axis movement decision
        else:
            self.yaw_velocity = 0

        # Move the drone (x, z, y, yaw)
        self.tello.send_rc_control(0, self.z_velocity, 0, self.yaw_velocity)