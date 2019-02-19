import os
import re
import glob
import numpy as np
import os.path as path
from scipy import ndimage, misc
from keras.models import model_from_json

#IMAGE_PATH = 'D:\Jhonatan\Imágenes\Try'
IMAGE_PATH = 'D:\Jhonatan\Imágenes\ImagenesPositivas'
file_paths = glob.glob(path.join(IMAGE_PATH, '*.jpg'))

images = []

for root, dirnames, filenames in os.walk(IMAGE_PATH):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            image = ndimage.imread(filepath, mode="RGB")
            image_resized = misc.imresize(image, (64, 128))
            images.append(image_resized)
            #plt.imshow(image_resized)
            #plt.show()
images = np.asarray(images)

n_images = images.shape[0]
labels = np.zeros(n_images)
for i in range(n_images):
    filename = path.basename(file_paths[i])[0]
    labels[i] = int(filename[0])

TRAIN_TEST_SPLIT = 0

shuffled_indices = np.random.permutation(n_images)
test_indices = shuffled_indices[0:]

x_test = images[test_indices, :, :]
y_test = labels[test_indices]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_run-20181104012901.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

test_predictions = loaded_model.predict(x_test)
test_predictions = np.round(test_predictions)

import matplotlib.pyplot as plt
def visualize_incorrect_labels(x_data, y_real, y_predicted):
    count = 0
    figure = plt.figure()
    '''
    incorrect_label_indices = (y_real != y_predicted)
    y_real = y_real[incorrect_label_indices]
    y_predicted = y_predicted[incorrect_label_indices]
    x_data = x_data[incorrect_label_indices, :, :, :]
    '''
    maximum_square = np.ceil(np.sqrt(x_data.shape[0]))

    for i in range(x_data.shape[0]):
        count += 1
        figure.add_subplot(maximum_square, maximum_square, count)
        plt.imshow(x_data[i, :, :, :])
        plt.axis('off')
        plt.title("Predicted: " + str(int(y_predicted[i])) + ", Real: " + str(int(y_real[i])), fontsize=10)

    plt.show()

visualize_incorrect_labels(x_test, y_test, np.asarray(test_predictions).ravel())