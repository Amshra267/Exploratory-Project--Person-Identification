"""
This is the main code for single person tracking which includes the pipeline like

YOLO----> Person detected(Yes)---->KCF Tracking-------------------------------------------------------------------------
                  |                                                                                                     |
                  |                                                                                                     |
                MTCNN(Face-detection)---------> Face Embeddings(Facenet) ----------> Comparison of embeddings if present else add -------> These two parts working in 
                                                                                                                               sync

"""

import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from MTCNN import FaceDetector
from imutils.video import FPS
from yolov3_person import load_yolo, get_box_dimensions, detect_objects
import math
from single_track_utils import OPENCV_OBJECT_TRACKERS
import torch.nn.functional as F
import json
import os
import torch
from datetime import datetime

def Database_verification(embedding = None, threshold = 0.3):

    """
    Functions for Face Recognition
    
    Arguments:
        embedding - A 512 Dimensional vector outputted from Facenet
        threshold - minimum value for similarity if person needs to be matched else not in Database  

    """

   # print(embedding.shape)
    with open("Database.json") as f:
        database_dict = json.load(f)
    members = len(database_dict)  # Number of members already in the database
  #  print("Number of members already in database - {}".format(members))
    cosine_max = 0
    identity = None
    for member in database_dict:
        similarity = F.cosine_similarity(embedding.cuda(), torch.from_numpy(np.array(database_dict[str(member)])).cuda(), dim = 1).item()
        if cosine_max < similarity:
            cosine_max = similarity
            identity = member
  #  print(cosine_max)
    if cosine_max <= threshold:
        identity = "Person Not from Organization"
    
    curr_time = datetime.now()
    curr_time = curr_time.strftime("%d/%m/%Y %H:%M:%S")
    print(f"Detected - {identity} at - {curr_time}")
    print(identity)
    return identity      


def single_person(cap, model, output_layers, tracker_type):
    """
        Function to detect single person from a video stream based on largest area occupied person among all
    
    Arguments:
          
          cap - cv2.videoCapture Object
          model - yolo_model
          output_layers - layers of that yolo_model
          tracker_type - TRACKER TYPE USED out of all available
    Retuens:
          single person location along with other utilities
    """
    _, frame = cap.read()
    height, width, _ = frame.shape
    tracker = OPENCV_OBJECT_TRACKERS[tracker_type]()
    blob, outputs = detect_objects(frame, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    boxes = np.array(boxes)[indexes]
    fps = FPS().start()
    person = None
    if len(boxes)!=0:
        if len(np.array(boxes).shape)!=3:  # if only single box comes in non max suppression then it passes as 2D array, need to make it 3D
            boxes = np.expand_dims(boxes, axis=0)

        boxes = boxes.transpose(0,2,1)
        boxes = np.clip(boxes, 0, width)
        # print(boxes)
        boxes = sorted(boxes, key= lambda x: (x[2]*x[3]), reverse = True)
        target_box = tuple([int(v) for v in boxes[0]])
        person = frame[target_box[1]:target_box[1]+target_box[3], target_box[0]:target_box[0]+target_box[2]]
        tracker.init(frame, target_box)
    return tracker, height, frame, fps, person

def start_video(video_path, tracker_type = "kcf"):
    """
    Main function governing single tracking

    Arguments:
        video_path - path of video file 
        tracker_type = cv2. tracking method (default = "kcf")
    
    Retuens: 
        None
    """

    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    initBB = None #bounding box initially none

    cap = cv2.VideoCapture(video_path) 

    while True:
        if initBB:
            _, frame = cap.read()
            if frame is None:
                break
            #print(frame.shape)
            success, target_box = tracker.update(frame)
            fps.update()
            fps.stop()
            if success: ## if rect detected and updated successfully
                (x, y, w, h) = [int(v) for v in target_box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                tracker, height, frame, fps, person = single_person(cap, model, output_layers, tracker_type)
                if person is not None:
                    yield person
            # initialize the set of information we'll be displaying on
            # the frame
           # print(fps.fps())
            info = [
                ("Tracker", tracker_type),
                ("Success", "Yes" if success else "Alternative"),
                ("FPS", "{:.2f}".format(fps.fps() if success else 10)),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, height - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            
        else:
            initBB = True
            tracker, height, frame, fps, person = single_person(cap, model, output_layers, tracker_type)
            if person is not None:
               yield person

        # show the output frame
        if frame is not None:
            cv2.imshow("INPUT", frame)
        #draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    gen = start_video("sample_videos/test7.mp4")
    mtcnn = MTCNN(select_largest = True , device = 'cuda')
    # Create an inception resnet (in eval mode):
    reco = InceptionResnetV1(pretrained='vggface2').eval()
    while True:
        identity = None
        try:
            image = next(gen)
           # cv2.imshow("person", image)
            fcd = FaceDetector(mtcnn, reco, is_image = True, img = image)
            embedding = fcd.run()
            if embedding is not None: # if face detected
                identity = Database_verification(embedding)
            # else identity none
            cv2.putText(image, str(identity), (0, image.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Person", image)

        except Exception as E:
            print(E)
            break

