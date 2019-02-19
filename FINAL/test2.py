import os
import numpy as np
from scipy import ndimage, misc
from keras.models import model_from_json
import matplotlib.pyplot as plt
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def draw(x_test, model):
    labels = ['Mano abierta', 'Puño', 'Indice', 'Nada']

    x_test = x_test / 255.0
    x_test = [x_test, x_test]
    x_test = np.asarray(x_test)

    p = model.predict(x_test)
    indices = np.argmax(p, 1)
    print(labels[indices[0]])

    #plt.imshow(x_test[0])
    #plt.axis('off')
    #plt.title("Predicted: " + labels[indices[0]], fontsize=10)

    #plt.show()


def load_m():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    return loaded_model


def main():
    camera = cv2.VideoCapture(0)
    model = load_m()

    while True:
        _, frame = camera.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        kernel = np.ones((5, 5), np.uint8)
        ColorLow = np.array([17, 77, 126])
        ColorHigh = np.array([45, 255, 255])

        mask = cv2.inRange(hsv, ColorLow, ColorHigh)
        kernel = np.ones((3, 3), np.uint8)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.erode(mask, None, iterations=3)

        frame = cv2.merge((mask, mask, mask))
        #frame = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('frame', frame)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = misc.imresize(frame, (120, 160))
        draw(image, model)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


