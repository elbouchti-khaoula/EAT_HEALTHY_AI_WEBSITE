from source.utils import calories
from source.model import get_model
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
size = 128
nb = 8
b=1
test_data ='C:/Users/User/Desktop/test.jpg'
model_save_at = os.path.join("models", 'model_')
def imagedtct(test_data):
    model = get_model(size, nb, 1e-3)
    model.load(model_save_at)
    labels = list(np.load('C:/Users/User/Desktop/project/source/labels.npy'))

    img = cv2.imread(test_data, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(img, img)
    model_out = model.predict([img])
    result = np.argmax(model_out)
    # predict ->tflearn and argmax ->numpy
    name = labels[result]
    cal = round(calories(result + 1, img), 2)
    print(name, cal, "Kcal")
    cv2.putText(img, name+cal+ "Kcal" , (0, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 0, 0, 0), 1)
    return img


if __name__ == '__main__':
    img = imagedtct(test_data)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
