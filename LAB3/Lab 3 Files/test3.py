import os
import re
import glob
import numpy as np
import os.path as path
from scipy import ndimage, misc
from keras.models import model_from_json
from matplotlib import pyplot as plt

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

xS, yS = 540, 960#2007, 2178 #540, 960
#img = ndimage.imread("C1.png", mode="RGB")
img = ndimage.imread("G1A11.jpg", mode="RGB")
#img = ndimage.imread("G3A20.jpg", mode="RGB")
#img = ndimage.imread("G1A3.jpg", mode="RGB")
img_resized = misc.imresize(img, (xS, yS))

#img1 = ndimage.imread("T1.png", mode="L")
img1 = ndimage.imread("G1A11_0.jpg", mode="L")
img1_resized = misc.imresize(img1, (xS, yS))

k, n = 0, (xS - 15) * (yS - 15)
images = []
target = np.zeros(n)

for h in range(xS - 15):
    for w in range(yS - 15):
        crop = img_resized[h:h + 15, w:w + 15]
        crop = crop.ravel()
        target[k] = img1_resized[h + 7, w + 7]
        images.append(crop)
        k += 1

images = np.asarray(images)

images, target = images / 255, target / 255
x_test = images
y_test = target

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_test = sc.fit_transform(x_test)

json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("try1.h5")
print("Loaded model from disk")

#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(x_test, y_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

test_predictions = loaded_model.predict(x_test)
test_predictions = np.round(test_predictions)

B = np.reshape(test_predictions, (xS - 15, yS - 15))#(531, 951))
plt.imshow(B, cmap="gray")
plt.show()