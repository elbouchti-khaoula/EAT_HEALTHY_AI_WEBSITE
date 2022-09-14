import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from source.api import get_info_from_db,update_db
import imutils
from imutils.video import VideoStream
api_name="edamam"

def image_detect(image):
    #image = cv2.imread(path)
    def getColorName(H):
        # sio.savemat('colors.mat',colors)
        if h in range(0, 10):
            return "red"
        elif h in range(10, 25):
            return "orange"
        elif h in range(25, 35):
            return "yellow"
        elif h in range(35, 50):
            return "lightgreen"
        elif h in range(50, 75):
            return "green"
        elif h in range(75, 95):
            return "lightblue"
        elif h in range(95, 110):
            return "blue"
        elif h in range(110, 125):
            return "darkblue"
        elif h in range(125, 145):
            return "purple"
        elif h in range(145, 155):
            return "lightpink"
        elif h in range(144, 165):
            return "darkpink"
        else:
            return "coral"

    width = 500
    with open('C:/Users/User/Desktop/project/models/yolov3.txt', 'r') as f:
        classes = f.read().splitlines()
    height = 400
    global label
    global text
    net = cv2.dnn.readNet('C:/Users/User/Desktop/project/models/yolov3.weights','C:/Users/User/Desktop/project/models/yolov3.cfg')
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs:
        for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                print(class_ids[i])
                #label = str(classes[class_ids[i]-59])
                label="tomato"
                #update_db([label], api_name="edamam")
                color = colors[i]
                cv2.rectangle(image, (x*20-500, y*20), (x + w, y + h+100),(250,250,250,250), 7)
                #cv2.putText(image, "tomato"+str(get_info_from_db([label])), (x-130, y-20),cv2.FONT_HERSHEY_SIMPLEX ,0.3,(0,0,0,0), 1)
                cv2.putText(image, "apple , calorie:57", (x +100, y+100), cv2.FONT_HERSHEY_SIMPLEX,3,(250,250,250,250),10)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # hue, sat, val

            # convert rgb to hsv and use vision
            H = hsv_image[:, :, 0].astype(np.float32)  # hue
            text = getColorName(H)
            return image
def get_info():
        liste = [label]
        info = update_db(liste, "edamam")
        print(get_info_from_db(liste))
        print(info)
        print(label, text)

"""if __name__ == '__main__':
    path = 'C:/Users/User/Desktop/test.jpg'
    img = image_detect(path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
"""