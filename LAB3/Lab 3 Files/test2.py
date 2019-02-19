import os
import re
import glob
import numpy as np
import os.path as path
from scipy import ndimage, misc
from keras.models import model_from_json
from matplotlib import pyplot as plt
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Sequential
import cv2

import tensorflow as tf

#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)

#xS, yS = 900, 900#2007, 2178 #540, 960
#img = ndimage.imread("SC.png", mode="RGB")
#img = ndimage.imread("G1A11.jpg", mode="RGB")
#img = ndimage.imread("G1A3.jpg", mode="RGB")
#img = ndimage.imread("G1A9.jpg", mode="RGB")
img = ndimage.imread("G3A20.jpg", mode="RGB")
xA, yA, _ = img.shape
xS, yS = int(xA), int(yA)
img_resized = misc.imresize(img, (xS, yS))

#img1 = ndimage.imread("T1.png", mode="L")
img1 = ndimage.imread("G1A11_0.jpg", mode="L")
img1_resized = misc.imresize(img1, (xS, yS))

r = 15
k, n, side = 0, (xS - r) * (yS - r), (r - 1) // 2
images = []
target = np.zeros(n)

for h in range(xS - r):
    for w in range(yS - r):
        crop = img_resized[h:h + r, w:w + r]
        target[k] = img1_resized[h + side, w + side]
        images.append(crop)
        k += 1

images = np.asarray(images)

image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
print(image_size)

images, target = images / 255, target / 255
x_test = images
y_test = target

json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("try1.h5")
#loaded_model.load_weights("weights00000005.h5")
print("Loaded model from disk")

#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(x_test, y_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

test_predictions = loaded_model.predict(x_test)
test_predictions = np.round(test_predictions)

B = np.reshape(test_predictions, (xS - r, yS - r))#(531, 951))

kernel = np.ones((11, 11), np.uint8)
opening = cv2.morphologyEx(B, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

plt.imshow(closing, cmap="gray")
plt.show()