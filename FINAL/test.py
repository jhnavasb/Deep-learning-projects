import os
import re
import glob
import numpy as np
import os.path as path
from scipy import ndimage, misc
from keras.models import model_from_json
import matplotlib.pyplot as plt


IMAGE_PATH = 'D:\Jhonatan\Documentos\DL\FINAL\DataManos'
#file_paths = glob.glob(path.join(IMAGE_PATH, '*.png'))
file_paths = []
for ext in ('*.jpg', '*.png'):
    file_paths.extend(glob.glob(path.join(IMAGE_PATH, ext)))

images = []

for root, dirnames, filenames in os.walk(IMAGE_PATH):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            image = ndimage.imread(filepath, mode="RGB")
            image_resized = misc.imresize(image, (120, 160))
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

loaded_model.load_weights("model.h5")
print("Loaded model from disk")

test_predictions = loaded_model.predict(x_test)
test_predictions = np.round(test_predictions)

labels = ['Mano abierta', 'Puño', 'Indice', 'Nada']

count = 0
figure = plt.figure()
maximum_square = np.ceil(np.sqrt(x_test[:16].shape[0]))

indices = np.argmax(loaded_model.predict(x_test[:16]), 1)

for i in range(x_test[:16].shape[0]):
    count += 1
    figure.add_subplot(maximum_square, maximum_square, count)
    plt.imshow(x_test[:16][i, :, :, :])
    plt.axis('off')
    plt.title("Predicted: " + labels[indices[i]], fontsize=10)

plt.show()

