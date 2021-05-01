"""
File used for inferencing of yolo using cv2.dnn Module

"""


import cv2
import numpy as np

def load_yolo():
    net = cv2.dnn.readNet("yolo_cnfgs/yolov3.weights", "yolo_cnfgs/yolov3.cfg")
  #  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
   # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    classes = []
    with open("yolo_cnfgs/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    classes = [0]
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            #print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if class_id in classes:
                if conf > 0.3:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.6, 0.5)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
           # print(x, y, w, h)
            imgs = img[y:y+h, x:x+w]
          #  if imgs.shape[1]>0 and imgs.shape[0]>0:
                #cv2.imshow("imgs"+str(i), imgs)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
            cv2.putText(img, str(round(confs[i], 2)), (x+w, y - 5), font, 1, color, 1)
    img = cv2.resize(img, (1080, 720))
    cv2.imshow("Image", img)

def start_video(video_path):
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_video("sample_videos/test1.mp4")
