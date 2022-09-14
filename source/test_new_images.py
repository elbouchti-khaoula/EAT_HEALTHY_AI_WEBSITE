from source.model import get_model
import matplotlib.pyplot as plt
import os
import imutils
import cv2
import numpy as np
import tensorflow
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from keras_preprocessing.image import img_to_array
size = 128
nb = 8
b=1
POSSIBLE_EXT = [".png", ".jpg", ".jpeg"]
model = tensorflow.keras.models.load_model('C:/Users/User/Desktop/project/models/food_mobilenet.h5')
"""def preprocess_loadimg_frame(loadimg_frame):
    # convert to RGB
    loadimg_frame = cv2.cvtColor(loadimg_frame, cv2.COLOR_BGR2RGB)
    # preprocess input image for mobilenet
    loadimg_frame_resized = cv2.resize(loadimg_frame, (224, 224))
    loadimg_frame_array = img_to_array(loadimg_frame_resized)
    return loadimg_frame_array"""
def decode_prediction(pred):
    lab = list(np.load('C:/Users/User/Desktop/project/source/labels.npy'))
    labels=tuple(lab)
    labels = pred
    item = max(labels)
    confidence = f"{(max(labels) * 100):.2f}"
    return item, confidence
def write_bb(item, confidence, box, frame):
    (x, y, w, h) = box
    color = (0, 255, 0)
    label = f"{item,}: {confidence}%"
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def load_cascade_detector():
    cascade_path = os.path.dirname(cv2.__file__) + "/image/classifier/cascade.xml"
    loadimg_detector = cv2.CascadeClassifier(cascade_path)
    return loadimg_detector

loadimg_detector_model = load_cascade_detector()

def imagedtct(image):
    image = imutils.resize(image, width=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadimg = loadimg_detector_model.detectMultiScale(gray,
                                                 scaleFactor=1.05,
                                                 minNeighbors=4,
                                                 minSize=(40, 40),
                                                 flags=cv2.CASCADE_SCALE_IMAGE,
                                                 )
    clone_image = image.copy()
    loadimg_dict = {"loadimg_list": [],
                  "loadimg_rect": []
                  }
    for rect in loadimg:
        (x, y, w, h) = rect
        loadimg_frame = clone_image[y:y + h, x:x + w]
        # preprocess image
        loadimg_frame_array = preprocess_loadimg_frame(loadimg_frame)

        loadimg_dict["loadimg_list"].append(loadimg_frame_array)
        loadimg_dict["loadimg_rect"].append(rect)

    if loadimg_dict["loadimg_list"]:
        loadimg_preprocessed = preprocess_input(np.array(loadimg_dict["loadimg_list"]))
        preds = model.predict(loadimg_preprocessed)
        for i, pred in enumerate(preds):
            item, confidence = decode_prediction(pred)
            write_bb(item, confidence, loadimg_dict["loadimg_rect"][i], clone_image)

    return clone_image
def test_on_custom_image(path):
    filename, file_extension = os.path.splitext(path)
    if file_extension not in POSSIBLE_EXT:
        raise Exception("possible file extensions are .png, .jpg, .jpeg")
    if not os.path.exists(path):
        raise FileNotFoundError("file not found")
    image = cv2.imread(path)
    image_masked = imagedtct(image)
    return image_masked


if __name__ == '__main__':
    test_data = 'C:/Users/User/Desktop/test.jpg'
    img = test_on_custom_image(test_data)
    plt.imshow(img)

