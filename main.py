'''
In this file is the main function for the project.
Simplified Algorithm:
 -The drone object recieves a frame and returns it to main.
 -That frame is then preprocessed and ran through yolov4-tiny(416).
 -That frame is also prerocessed and ran through gesture control.
 -The results are returned from yolo, then transformed and cleaned.
 -The transformed data is then ran through deepsort. 
 -Deepsort returns the tracked data to main.
 -Gesture control returns the correct drone movement to main.
 -The drone is then sent the gesture movement commands. (moves)
 -The drone is then sent the tracked data, and decides to moves or not.
 -Repeat. 

During the early stages of the project I used many tutorials and guides
to aid my understanding of yolo and other compelx algorithms. 
Parts of this file are taken from a guide by 'The AI Guy'. 
https://www.youtube.com/watch?v=FuvQ8Melz1o&t=483s
'''

# Tensorflow imports
import tensorflow as tf

# yolo utility imports
import core.utils as utils
from core.config import cfg

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from utils import generate_detections as gdet

# drone imports
from drone.drone import Drone

# gesture imports
from gesture.control import GestureControl

# utility imports
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
import os.path
from threading import Thread

def main():   
    # create drone object
    drone = Drone()
    drone_takeoff = False

    # create gesture control object
    gesture_control_on = False
    if gesture_control_on:
        gesture = GestureControl()
        gesture_result = ''

    # definition of the parameters, used for 
    # NearestNeighborDistanceMetric in the deepsort library
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize NNetwork for DeepSort
    # calculate cosine distance metric
    # initialize tracker
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # load yolov4 tiny as object detection NNetwork //////////
    input_size = 416
    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-tiny-416')
    infer = saved_model_loaded.signatures['serving_default']

    # Video Loop
    frame_num = 0
    while True:

        # get frame recieved from drone
        # transform the frame to the format that yolo requires as an input ///////////////////
        frame = drone.get_frame() 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        frame_num +=1
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # send frame to gesture control, then send the result to the drone.
        if gesture_control_on:
            gesture.set_frame(frame)
            gesture_result = gesture.get_gesture_prediction()
            drone.gesture_control(gesture_result)

        # get class predictions and bounding boxes (data from yolo) ////////
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.5,      # intersection of boxes
            score_threshold=0.7     # prediction threshold
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format the bounding boxes ready for cv2
        # store all predictions in one parameter for simplicity when calling functions
        # read in all class names from config
        # custom allowed classes
        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        pred_bbox = [bboxes, scores, classes, num_objects]
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        allowed_classes = ['person'] 
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        # delete detections that we don't need
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # call the tracker
        tracker.predict()
        tracker.update(detections)   

        # flag, "has the user been tracked this frame?"
        tracked = False                           

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed():
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
            # draw bbox on screen
            if track.time_since_update < 5:
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # decide if this track is the user
            # if so, drone makes a decision and moves
            if tracked == False and class_name == 'person' and track.time_since_update < 30:
                drone.make_movement_decision(bbox, frame)
                tracked = True
            
        # Launch the drone and video thread, if gestures are not on
        if not gesture_control_on:
            if drone_takeoff == False:
                drone.takeoff()
                drone_takeoff = True
                drone.start_video_thread()

        # calculate frames per second of running detections
        # convert gbr to rgb, transformation for viewing
        fps = 1.0 / (time.time() - start_time)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # display variable statuses to screen
        battery_string = 'BATTERY: ' + str(drone.get_battery())
        cv2.putText(result, battery_string, (750, 40), font, 1, (204, 0, 0), 2, cv2.LINE_AA)
        fps_string = 'FPS: ' + str(int(fps))
        cv2.putText(result, fps_string, (800, 80), font, 1, (0, 102, 34), 2, cv2.LINE_AA)
        if gesture_control_on:
            gesture_string = 'GESTURE: ' + str(gesture_result)
            cv2.putText(result, gesture_string, (600, 120), font, 1, (204, 0, 0), 2, cv2.LINE_AA)
        z_velocity_string = 'Z VELOCITY: ' + str(drone.z_velocity)
        cv2.putText(result, z_velocity_string, (40, 40), font, 1, (0, 0, 153), 2, cv2.LINE_AA)
        yaw_velocity_string = 'YAW VELOCITY: ' + str(drone.yaw_velocity)
        cv2.putText(result, yaw_velocity_string, (40, 80), font, 1, (0, 0, 153), 2, cv2.LINE_AA)

        # States to start cv2.imshow
        if gesture_control_on:
            if drone.tello_takeoff:
                cv2.imshow("Drone Vision", result)
        else:
            cv2.imshow("Drone Vision", result)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # close the viewing window
    cv2.destroyAllWindows()

    # join the gesture thread if gestures are enabled
    if gesture_control_on:
        gesture.join_thread()

if __name__ == '__main__':
    main()
