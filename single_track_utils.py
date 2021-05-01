"""
It is just a fun implementation of using different tracking algorithm implemented in OpenCV on ROI image 
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils.video import VideoStream
from imutils.video import FPS
import imutils

## DiFFERENT TPES OF OpenCV Object Trackers
OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}


    
if __name__ == "__main__":

    tracker = OPENCV_OBJECT_TRACKERS["kcf"]()
    initBB = None #bounding box initially none

    cap = cv2.VideoCapture("sample_videos/test1.mp4") 
    fps = None
    while(True):
        ret, frame = cap.read()
        (H, W) = frame.shape[:2]
        if frame is None:
            break
        if initBB is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)

            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)

            # update the FPS counter
            fps.update()
            fps.stop()

            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", "kcf"),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(10) & 0xFF
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                showCrosshair=True)

            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()