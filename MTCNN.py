"""
Function containing implemtation for functions for using in original main script ana also for fun in webcam mode
"""


import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn, reco, is_image = False, img = None):
        self.mtcnn = mtcnn
        self.reco = reco
        self.is_image = is_image
        self.img = img
    def _draw(self, frame, box, prob):
        """
        Draw landmarks and boxes for each face detected
        """
        try:
            # Draw rectangle on frame
            cv2.rectangle(frame,
                            (int(box[0][0]), int(box[0][1])),
                            (int(box[0][2]), int(box[0][3])),
                            (0, 0, 255),
                            thickness=2)

                # Show probability
                # cv2.putText(frame, str(
                #     prob), (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw landmarks
            # cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)
        except Exception as e:
        #    print(e)
            pass
 
    
    def action(self, image):
         # detect face box, probability and landmarks
        boxes, probs = self.mtcnn.detect(image, landmarks=False)
       # print(len(boxes))
        with torch.no_grad():
            embedding = self.reco(self.mtcnn(image).unsqueeze(0))
         
        #print(embedding.shape)
        plt.show()
        # print(boxes)
        # draw on frame
        self._draw(image, boxes, probs)
        return embedding
    def run(self):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """
        embedding = None
        if not self.is_image:
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()

                try:
                    self.action(frame)
                except:
                    pass

                # Show the frame
                cv2.imshow('Face Detection', frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            try:
                embedding = self.action(self.img)
                cv2.imshow('Face Detection', self.img)
            except Exception as e:
              #  print(e)
                pass

            # Show the frame
        return embedding
        
# Run the cam

if __name__ == "__main__":
    mtcnn = MTCNN(device = 'cuda', keep_all = False)
    # Create an inception resnet (in eval mode):
    reco = InceptionResnetV1(pretrained='vggface2').eval()

    fcd = FaceDetector(mtcnn, reco, is_image = False, img = None)
    fcd.run()